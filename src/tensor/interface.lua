local interface = {}

local function bit(p)
   return 2 ^ (p - 1)  -- 1-based indexing
end

local function hasbit(x, p)
   return x % (p + p) >= p
end

local function tabledup(tbl)
   local newtbl = {}
   for k,v in pairs(tbl) do
      if type(v) == 'table' then
         local mt = getmetatable(v)
         v = tabledup(v)
         if mt then
            setmetatable(v, mt)
         end
      end
      newtbl[k] = v
   end
   return newtbl
end

local function tablesortbykey(tbl)
   local newtbl = {}
   for k,v in pairs(tbl) do
      table.insert(newtbl, {k=k, v=v})
   end
   table.sort(newtbl, function(a, b) return a.k < b.k end)
   return newtbl
end

local types = dofile('types.lua')
local function maketype(tbl)
   local name = tbl.name
   name = name:gsub('^%^+', '')
   local mtbl = types.mt[name]
   assert(mtbl, string.format('unknown type <%s>', tbl.name))
   setmetatable(tbl, {__index=mtbl})
   return tbl
end

local function makecarg(args)
   local function cclosure(idx)
      return function() return string.format("carg%d", idx) end
   end
   local function ccclosure(idx)
      return function() return string.format("ccarg%d", idx) end
   end
   for idx,arg in ipairs(args) do
      arg.carg = cclosure(idx)
      arg.ccarg = ccclosure(idx)
   end
end

local signatures = {}

local function signature(name, ccargs, verbose)
   local signature = {}
   for _, retarg in ipairs(ccargs[0]) do
      table.insert(signature, retarg:signature())
   end
   table.insert(signature, name)
   if verbose then
      table.insert(signature, '(')
   end
   for idx,arg in ipairs(ccargs) do
      table.insert(signature, arg:signature())
   end
   if verbose then
      table.insert(signature, ')')
   end
   signature = table.concat(signature, verbose and ' ' or '_')
   return signature
end

local function wrap(name, cname, args)
   cname = cname:gsub('%^+', '') -- debug hack
   makecarg(args)

   -- hack: dynamic types
   for idx=1,#args do
      local arg = args[idx]
      if arg.name:match('^%^%^') then
         arg.dyn = 'accreal'
      elseif arg.name:match('^%^') then
         arg.dyn = 'real'
      end
      arg.name = arg.name:gsub('%^+', '')
      args[idx] = maketype(arg)
   end

   local nopt = 0
   local defgroups = {}
   for idx, arg in ipairs(args) do
      if not arg.invisible and arg.default ~= nil then
         if arg.defgroup then
            if not defgroups[arg.defgroup] then
               defgroups[arg.defgroup] = true
               nopt = nopt + 1
            end
         else
            nopt = nopt + 1
         end
      end
   end

   for variant=0,(2^nopt)-1 do
      local ccargs = {[0]={}}
      local dynidx

      local function checkreturn(arg)
         if arg.creturned or arg.returned then
            table.insert(ccargs[0], arg)
         end
      end
      -- dynidx refers to ccargs idx
      local function checkdyn(arg, idx)
         if arg.dyn then
            dynidx = dynidx or idx -- first ccargs which is dyn
         end
      end

      local opt = 0
      local defgroup
      local defgroups = {}
      for idx,arg in ipairs(args) do
         if arg.invisible then
            checkreturn(arg)
            -- nothing do be done for now
         elseif arg.default ~= nil then
            if arg.defgroup then
               if not defgroups[arg.defgroup] then
                  defgroups[arg.defgroup] = true
                  opt = opt + 1
                  if hasbit(variant, bit(opt)) then
                     defgroup = arg.defgroup
                  end
               end
            else
               opt = opt + 1
            end

            if (arg.defgroup and defgroup == arg.defgroup) or
            (not arg.defgroup and hasbit(variant, bit(opt))) then
               table.insert(ccargs, arg)
               checkdyn(arg, #ccargs) -- not idx
            else
               checkreturn(arg) -- do not return something which is a arg
            end
         elseif arg.creturned then
            checkreturn(arg)
         else
            table.insert(ccargs, arg)
            checkdyn(arg, #ccargs) -- not idx
            -- checkreturn(arg)
         end
      end
      local funcname = name
      if #ccargs[0] == 0 then
         table.insert(ccargs[0], maketype({name='void'}))
         funcname = funcname .. '_' -- remove some ambiguities
      end
      local signature = signature(funcname, ccargs)
      local signtensor = signature:gsub('real', 'Tensor')
      signtensor = signtensor:gsub('TensorD%d+', 'Tensor')
      local device = types.inferdevice(cname)
      if not signatures[signtensor] then
         signatures[signtensor] = {name=funcname}
         assert(signatures[signtensor])
      end
--      print(signature, dynidx)
      table.insert(signatures[signtensor], {cname=cname, device=device, funcname=funcname, ccargs=ccargs, args=args, variant=variant, signature=signature, dynidx=dynidx})
   end
end

function interface.wrap(name, ...)
   assert(select('#', ...) % 2 == 0)
   for i=1,select('#', ...)/2 do
      local cname = select(2*(i-1)+1, ...)
      local args = select(2*(i-1)+2, ...)
      wrap(name, cname, args)
   end
end

local function decl(prefix, name, ccargs)
   local decl = {}
   for _, ccarg in ipairs(ccargs) do
      assert(ccarg.decl, string.format("decl undefined for type <%s>", ccarg.name))
      table.insert(decl, ccarg:decl())
   end
   local returns
   if #ccargs[0] == 1 then
      returns = ccargs[0][1]:declreturn()
   else
      returns = {}
      for _, ret in ipairs(ccargs[0]) do
         table.insert(returns, ret:declreturn())
      end
      returns = string.format("std::tuple<%s>", table.concat(returns, ", "))
   end
   return string.format(
      "%s%s %s(%s)",
      prefix and (prefix .. " ") or "",
      returns,
      name and name or "",
      table.concat(decl, ", ")
   )
end

local function implement(rec, name, def, prefix)
   local ccargs = def.ccargs

   rec:add(decl(prefix, name, ccargs))

   rec:add("{")
   rec:indent()
   -- {cname=cname, ccargs=ccargs, args=args, dynarg=dynarg, variant=variant}
   -- read arguments
   for idx,arg in ipairs(ccargs) do
      assert(arg.read, string.format("undef read for type <%s>", arg.name))
      rec:add(arg:read())
   end

   -- init default arguments
   local opt = 0
   local defgroup
   local defgroups = {}
   for idx,arg in ipairs(def.args) do
      if arg.invisible then
         assert(arg.default ~= nil, 'invisible has no default')
         rec:add(arg:readdefault(def))
      elseif arg.default ~= nil then
         if arg.defgroup then
            if not defgroups[arg.defgroup] then
               defgroups[arg.defgroup] = true
               opt = opt + 1
               if hasbit(def.variant, bit(opt)) then
                  defgroup = arg.defgroup
               end
            end
         else
            opt = opt + 1
         end
         if not ((arg.defgroup and defgroup == arg.defgroup) or
                 (not arg.defgroup and hasbit(def.variant, bit(opt)))) then
         -- opt = opt + 1
         -- if not hasbit(def.variant, bit(opt)) then
            assert(arg.readdefault, string.format("undef readdefault for type <%s>", arg.name))
            rec:add(arg:readdefault(def))
         end
      end
   end

   -- precall
   for idx,arg in ipairs(def.args) do
      if arg.precall then
         rec:add(arg.precall(arg, def.args))
      end
   end

   -- create call
   local device = def.cname:match("Cuda") and "gpu" or "cpu"
   local callargs = {}
   local retarg
   for idx,arg in ipairs(def.args) do
      if arg.creturned then
         retarg = arg:readcreturned(def)
      else
         if not(arg.name == "Generator" and device == "gpu") then -- hack
            table.insert(callargs, arg:call())
         end
      end
   end
   rec:add(string.format("%s%s(%s%s);", retarg or "", def.cname, device == "gpu" and "thcstate(), " or "", table.concat(callargs, ", ")))

   -- postcall
   for idx,arg in ipairs(def.args) do
      if arg.postcall then
         rec:add(arg.postcall(arg, def.args))
      end
   end

   -- return
   local returns = {}
   for _, ret in ipairs(ccargs[0]) do
      if #ccargs[0] > 1 and ret.moveandreturnit then -- we will use std::tuple below
         ret = ret:moveandreturnit(rec)
      else
         ret = ret:returnit(rec)
      end
      if ret then
         table.insert(returns, ret)
      end
   end
   if #returns == 0 then
      returns = nil
   elseif #returns == 1 then
      returns = returns[1]
   else
      returns = string.format("std::make_tuple(%s)", table.concat(returns, ", "))
   end
   if returns then
      returns = string.format("return %s;", returns)
   end
   rec:add(returns)
   rec:unindent()
   rec:add("}")
end

local function isdyn(defs)
   if #defs > 1 then
      return true
   end
   local args = defs[1].args
   for _, arg in ipairs(args) do
      if arg.dyn then
         return true
      end
   end
   return false
end

local function implementnodyn(rec, signature, defs)
   assert(not isdyn(defs), 'function is dynamic; non-dynamic expected')
   local def = defs[1]
   implement(
      rec,
      signature,
      def,
      "static"
   )
end

local function implementdyn(rec, signature, defs)
   rec:add(string.format([[
struct %s_op
{
  template<typename T> %s { throw std::invalid_argument("%s: unsupported device cpu with type " + Tensor::typedesc<T>()); };
  template<typename T> %s { throw std::invalid_argument("%s: unsupported device gpu with type " + Tensor::typedesc<T>()); };
};
]], signature, decl(nil, "cpu", defs[1].ccargs), defs[1].funcname, decl(nil, "gpu", defs[1].ccargs), defs[1].funcname))
   for _,def in ipairs(defs) do
      local dyntype = types.inferdyntype(def)
      implement(
         rec,
         signature .. "_op::" .. def.device .. "<" .. dyntype .. ">",
         def,
         "template<>"
      )
   end
end

local function implementrt(rech, rec, defs)
   local signatures = {}
   for _, def in ipairs(defs) do
      local signature = def.signature
      if not signatures[signature] then
--         print(signature)
         signatures[signature] = {}
      end
--      signatures[signature] = signatures[signature] or {}
      table.insert(signatures[signature], def)
      assert(def.dynidx == signatures[signature][1].dynidx, 'dynamic arg mismatch?!')
   end
   signatures = tablesortbykey(signatures)
   if #signatures == 1 then
      if not isdyn(signatures[1].v) then
         implementnodyn(rec, signatures[1].k, signatures[1].v)
      else
         implementdyn(rec, signatures[1].k, signatures[1].v)
      end
   else
      for _, signature in ipairs(signatures) do
         if not isdyn(signature.v) then
            implementnodyn(rec, signature.k, signature.v)
         else
            implementdyn(rec, signature.k, signature.v)
         end
      end
--      error('NYI')
   end
   local args = {}
   -- everybody have the same tensor signature
   -- however beware than it might not match to same argument...
   local ref = signatures[1].v[1]
   for _,arg in ipairs(ref.ccargs) do
      table.insert(args, arg:ccarg())
      if arg.declinc then
         local declinc = arg:declinc()
         if declinc then
            rech:add(declinc)
         end
      end
   end
   args = table.concat(args, ", ")
   -- non-dynamic or dynamic arg is provided
   if not isdyn(signatures[1].v) or ref.dynidx then
      local def = decl(nil, defs.name, ref.ccargs)
      rech:add(def .. ";")
      rec:add(def)
   else -- dynamic: need to query explicitely the dynamic type
      local ccargs = tabledup(ref.ccargs)
      table.insert(ccargs, maketype({name='TensorType', ccarg=function() return "ttype" end}))
      table.insert(ccargs, maketype({name='TensorDevice', ccarg=function() return "tdev" end}))
      local def = decl(nil, defs.name, ccargs)
      rec:add(def)
      def = def:gsub("tdev", "tdev=kCPU");
      rech:add(def .. ";")
   end
   rec:add("{")
   rec:indent()
   local function dimccargs(ref)
      local dimargs = {}
      for id, ccarg in ipairs(ref.ccargs) do
         if ccarg:signature() == 'real' then
            table.insert(dimargs, {id=id, dim=0})
         elseif ccarg:signature():match('TensorD%d+') then
            table.insert(dimargs, {id=id, dim=ccarg.dim})
         end
      end
      return dimargs
   end
   table.sort(
      signatures,
      function(s1, s2)
         return #dimccargs(s1.v[1]) > #dimccargs(s2.v[1])
      end
   )
   -- DEBUG:
   if #signatures > 1 then
      for _, signature in ipairs(signatures) do
--         print('----', signature.v[1].signature)
      end
   end
   local needelse = false
   for signidx, signature in ipairs(signatures) do
      if #signatures > 1 then
         local dimas = dimccargs(signature.v[1])
         local cond = {}
         for _, dima in ipairs(dimas) do
            table.insert(cond, string.format("(%s.dim() == %d)", ref.ccargs[dima.id]:ccarg(), dima.dim)) -- beware: ref!! (see comment above)
         end
         if signidx > 1 then
            rec:unindent()
         end
         if #cond == 0 then
            rec:add("} else {")
            needelse = false
         else
            cond = string.format("%sif(%s) {", signidx == 1 and "" or "} else ", table.concat(cond, " && "))
            rec:add(cond)
            needelse = true
         end
         rec:indent()
      end
      if isdyn(signature.v) then
         local dynidx = signature.v[1].dynidx
         if dynidx then
            rec:add(string.format("return dispatch<%s_op>(%s);", signature.k, args))
         else
            rec:add(string.format("return dispatch<%s_op>(ttype, tdev, %s);", signature.k, args))
         end
      else
         rec:add(string.format("return %s(%s);", signature.k, args))
      end
   end
   if #signatures > 1 then
      if needelse then
         rec:unindent()
         rec:add("} else {")
         rec:indent()
         rec:add('throw std::invalid_argument("no matching function");')
      end
      rec:unindent()
      rec:add("}")
   end
   rec:unindent()
   rec:add("}")

end

function interface.implement(rech, rec)
   for _, signature in ipairs(tablesortbykey(signatures)) do
      implementrt(rech, rec, signature.v)
   end
end

function interface.write(rec, txt)
   rec:add(txt)
end

return interface

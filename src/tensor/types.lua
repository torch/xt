local types = {mt={}}
local mt = types.mt

local function setnyi(mtbl, name)
   setmetatable(
      mtbl,
      {
         __index =
            function(self, key)
               print(key, name)
               error("nyi: <%s> for type <%s>", key, name)
            end
      }
   )
end

function types.inferdevice(name)
   return name:match('Cuda') and 'gpu' or 'cpu'
end

local TensorType = {}
types.mt.TensorType = TensorType
function TensorType:decl()
   return string.format("TensorType %s", self:ccarg())
end
function TensorType:read()
   error('NYI')
end
function TensorType:signature()
   return "TensorType"
end

local TensorDevice = {}
types.mt.TensorDevice = TensorDevice
function TensorDevice:decl()
   return string.format("TensorDevice %s", self:ccarg())
end
function TensorDevice:read()
   error('NYI')
end
function TensorDevice:signature()
   return "TensorDevice"
end

local void = {}
types.mt.void = void
function void:decl()
   error('nyi')
end

function void:declreturn()
   return 'void'
end

function void:signature()
   return 'void'
end

function void:returnit()
end

local ttypes = {
   {name='Byte', base='uint8_t', xt='kUInt8'},
   {name='Char', base='int8_t', xt='kInt8'},
   {name='Short', base='int16_t', xt='kInt16'},
   {name='Int', base='int32_t', xt='kInt32'},
   {name='Long', base='int64_t', xt='kInt64'},
   {name='Float', base='float', xt='kFloat'},
   {name='Double', base='double', xt='kDouble'}
}
types.TensorTypes = ttypes

local ntypes = {
   {name='unsigned char', base='uint8_t', accreal='int64_t'},
   {name='char', base='int8_t', accreal='int64_t'},
   {name='short', base='int16_t', accreal='int64_t'},
   {name='int', base='int32_t', accreal='int64_t'},
   {name='long', base='int64_t', accreal='int64_t'},
   {name='float', base='float', accreal='double'},
   {name='double', base='double', accreal='double'}
}
types.NumberTypes = ntypes

function types.tensorbase(name)
   name = name:gsub('CudaTensor', 'CudaFloatTensor')
   name = name:gsub('Cuda', '')
   for _, t in ipairs(ttypes) do
      if name == t.name .. "Tensor" then
         return t.base
      end
   end
   error(string.format("unknown tensor type <%s>", name))
end

function types.tensorxt(name)
   for _, t in ipairs(ttypes) do
      if name == t.name .. "Tensor" then
         return t.xt
      end
   end
   error(string.format("unknown tensor type <%s>", name))
end

function types.base2ktensor(dyntype)
   for _, ttype in ipairs(ttypes) do
      if ttype.base == dyntype then
         return ttype.xt
      end
   end
   error(string.format("unknown dyntype <%s>", dyntype))
end

function types.inferdyntype(def)
   local name = def.cname
   name = name:gsub('CudaTensor', 'CudaFloatTensor')
   name = name:gsub('Cuda', '')
   name = name:match("^TH(.*)Tensor_.*")
   if name then
      for k,v in ipairs(ttypes) do
         if v.name == name then
            return v.base
         end
      end
      error(string.format("unknown torch type <%s>", name))
   end
   local args = def.args
   for _, arg in ipairs(args) do
      local name = arg.name
      for k,v in ipairs(ntypes) do
         if v.name == name then
            return v.base
         end
      end
   end
   error('could not infer dyntype')
end

for _, device in ipairs{"cpu", "gpu"} do
   for _, ttype in ipairs(ttypes) do
      local Tensor = {}
      local subname = (device == "cpu" and "" or "Cuda") .. ttype.name
      if subname == "CudaFloat" then
         subname = "Cuda"
      end
      types.mt[subname .. "Tensor"] = Tensor
      function Tensor:decl()
         return string.format("%sTensor& %s", (self.returned or self.notconst) and "" or "const ", self:ccarg())
      end
      function Tensor:declreturn()
         return string.format("Tensor")
      end
      function Tensor:read()
         local txt = {}
         table.insert(txt, string.format("TH%sTensor *%s = %s.THTensor<TH%sTensor>();", subname, self:carg(), self:ccarg(), subname))
         if self.dim then
            table.insert(txt, string.format('if(TH%sTensor_nDimension(%s%s) != %s) { throw std::invalid_argument("%d-dim tensor expected"); }', subname, device == "cpu" and "" or "thcstate(), ", self:carg(), self.dim, self.dim))
         end
         return table.concat(txt, "\n");
      end
      function Tensor:call()
         return string.format("%s", self:carg())
      end
      function Tensor:readdefault(ctx)
         if type(self.default) == 'boolean' then
            return string.format("Tensor %s(%s, k%s); TH%sTensor *%s = %s.THTensor<TH%sTensor>();", self:ccarg(), ttype.xt, device:upper(), subname, self:carg(), self:ccarg(), subname)
         elseif type(self.default) == 'number' then
            return string.format("TH%sTensor *%s = %s.THTensor<TH%sTensor>();", subname, self:carg(), ctx.args[self.default]:ccarg(), subname)
         else
            error('NYI')
         end
      end
      function Tensor:returnit()
         return self:ccarg()
      end
      function Tensor:moveandreturnit()
         return string.format("std::move(%s)", self:ccarg())
      end
      function Tensor:signature()
         if self.dyn then
            return string.format("Tensor%s", self.dim and ("D" .. self.dim) or "")
         else
--DEBUG:            return subname .. string.format("Tensor%s", self.dim and ("D" .. self.dim) or "")
            return string.format("Tensor%s", self.dim and ("D" .. self.dim) or "")
         end
      end
   end
end

local LongArg = {}
types.mt.LongArg = LongArg

function LongArg:decl()
   return string.format("std::vector<int64_t> %s", self:ccarg())
end

function LongArg:read()
   local txt = {}
--   std::unique_ptr<THLongStorage>
   table.insert(txt, string.format("auto %s = std::unique_ptr<THLongStorage, std::function<void (THLongStorage*)>>(THLongStorage_newWithSize(%s.size()), THLongStorage_free);", self:carg(), self:ccarg()));
   table.insert(txt, string.format("for(uint64_t i = 0; i < %s.size(); i++) { %s->data[i] = %s[i]; };", self:ccarg(), self:carg(), self:ccarg()))
   return table.concat(txt, "\n")
end

function LongArg:call()
   return string.format("%s.get()", self:carg())
end

function LongArg:signature()
   return "LongArg"
end

local function base2acc(base, dyn)
   if not dyn or dyn == 'real' then
      return base
   end
   for _, ntype in ipairs(ntypes) do
      if ntype.base == base then
         return ntype.accreal
      end
   end
   error(string.format("unknown base type <%s>", base))
end

for _, ntype in ipairs(ntypes) do
   local number = {}
   types.mt[ntype.name] = number
--   setnyi(number, ntype.name)
   function number:decl()
      if self.dyn then
         return string.format("const Tensor& %s", self:ccarg()) -- const: a value is indeed never changed
      else
         return string.format("%s %s", ntype.base, self:ccarg())
      end
   end
   function number:declreturn()
      if self.dyn then
         return string.format("Tensor")
      else
         return string.format("%s", ntype.base)
      end
   end
   function number:read()
      if self.dyn then
         return string.format("%s %s = %s.value<%s>();", base2acc(ntype.base, self.dyn), self:carg(), self:ccarg(), base2acc(ntype.base, self.dyn))
      else
         return string.format("%s %s = %s;", ntype.base, self:carg(), self:ccarg())
      end
   end
   function number:readcreturned()
      -- if self.dyn then
      --    return string.format("Tensor %s; %s.value<%s>() = ", self:ccarg(), self:ccarg(), base2acc(ntype.base))
      -- else
      return string.format("%s %s = ", ntype.base, self:carg())
      -- end
   end
   function number:readdefault(ctx)
      if type(self.default) == 'number' then
--          if self.dyn then
-- --            return string.format("Tensor %s; %s.value<%s>() = %s; %s %s = ", self:ccarg(), self:ccarg(), base2acc(ntype.base), self.default)
--             return string.format("%s %s = %s;", base2acc(ntype.base), self:carg(), self.default)
--          else
         return string.format("%s %s = %s;", base2acc(ntype.base, self.dyn), self:carg(), self.default)
--         end
      elseif type(self.default) == 'function' then
         return string.format("%s %s = %s;", base2acc(ntype.base, self.dyn), self:carg(), self.default(ctx))
      else
         error('NYI')
      end
   end
   function number:call()
      return self:carg()
   end
   function number:signature()
      if self.dyn then
         return "real"
      else
         return ntype.name
      end
   end
   function number:returnit(rec)
      if self.dyn then
         rec:add(string.format("Tensor %s((%s)%s);", self:ccarg(), base2acc(ntype.base, self.dyn), self:carg()))
         return self:ccarg()
      else
         return self:carg()
      end
   end
end

local enums = {}
local boolean = {}
local ntype = {name='boolean', base='bool'}
types.mt.boolean = boolean
function boolean:declinc()
   if self.enum then
      if not enums[self.enum] then
         enums[self.enum] = true
         return string.format("enum %s {%s, %s};", self.enum.name, self.enum[1], self.enum[2])
      end
   end
end
function boolean:decl()
   assert(not self.dyn, 'NYI')
   return string.format("%s %s", self.enum and self.enum.name or ntype.base, self:ccarg())
end
function boolean:declreturn()
   assert(not self.dyn, 'NYI')
   return string.format("%s", self.enum and self.enum.name or ntype.base)
end
function boolean:read()
   assert(not self.dyn, 'NYI')
   if self.enum then
      return string.format("%s %s = (%s == %s ? false : true);", ntype.base, self:carg(), self:ccarg(), self.enum[1])
   else
      return string.format("%s %s = %s;", ntype.base, self:carg(), self:ccarg())
   end
end
function boolean:readcreturned()
   return string.format("%s %s = ", ntype.base, self:carg())
end
function boolean:readdefault(ctx)
   if type(self.default) == 'boolean' then
      return string.format("%s %s = %s;", ntype.base, self:carg(), self.default)
   else
      error('NYI')
   end
end
function boolean:call()
   return self:carg()
end
function boolean:signature()
   assert(not self.dyn, 'NYI')
   return ntype.name
end
function boolean:returnit()
   assert(not self.dyn, 'NYI')
   if self.enum then
      return string.format("(%s ? %s : %s)", self:carg(), self.enum[2], self.enum[1])
   else
      return self:carg()
   end
end

local Generator = {}
types.mt.Generator = Generator
function Generator:decl()
   assert(not self.dyn, 'NYI')
   return string.format("Context &%s", self:ccarg())
end
function Generator:declreturn()
   error('NYI')
end
function Generator:read()
   assert(not self.dyn, 'NYI')
   return string.format("THGenerator *%s = %s.generator().get();", self:carg(), self:ccarg())
end
function Generator:readcreturned()
   error('NYI')
end
function Generator:readdefault(ctx)
   if self.default ~= nil then
      return string.format("THGenerator *%s = defaultContext.generator().get();", self:carg())
   end
end
function Generator:call()
   return self:carg()
end
function Generator:signature()
   assert(not self.dyn, 'NYI')
   return "Generator"
end
function Generator:returnit()
   error('NYI')
end

local charoption = {}
types.mt.charoption = charoption
function charoption:decl()
   return string.format("const char %s", self:ccarg())
end
function charoption:declreturn()
   error('NYI')
end
function charoption:read()
end
function charoption:readcreturned()
   error('NYI')
end
function charoption:readdefault(ctx)
   if self.default ~= nil then
      return string.format("const char %s = '%s';", self:ccarg(), self.default)
   end
end
function charoption:call()
   return string.format("&%s", self:ccarg())
end
function charoption:signature()
   return "charoption"
end
function charoption:returnit()
   error('NYI')
end

local THCState = {}
types.mt.THCState = THCState
function THCState:decl()
   error('nyi')
end

function THCState:readdefault(ctx)
   if self.default == true then
   else
      error('NYI')
   end
end

function THCState:call()
   return "thcstate()"
end

function THCState:declreturn()
   return 'THCState'
end

function THCState:signature()
   return 'THCState'
end

function THCState:returnit()
end

types.mt.index = types.mt.long

types.mt.IndexTensor = {}
for k,v in pairs(types.mt.LongTensor) do
   types.mt.IndexTensor[k] = v
end
function types.mt.IndexTensor:signature()
   assert(not self.dyn, 'NYI')
   return string.format("IndexTensor%s", self.dim and ("D" .. self.dim) or "")
end
types.mt.CudaIndexTensor = {}
for k,v in pairs(types.mt.CudaLongTensor) do
   types.mt.CudaIndexTensor[k] = v
end
function types.mt.CudaIndexTensor:signature()
   assert(not self.dyn, 'NYI')
   return string.format("IndexTensor%s", self.dim and ("D" .. self.dim) or "")
end

return types

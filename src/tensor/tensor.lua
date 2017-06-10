local interface = dofile('interface.lua')
local wrap = interface.wrap
local Record = dofile('record.lua')
local types = dofile('types.lua')

local rech = Record()
local rec = Record()

assert(#arg == 3, string.format("expected: %s <.h> <.cc> <true/false (cuda)>", arg[0]))

local devices
if arg[3]:lower() == 'true' then
   devices = {"cpu", "gpu"}
else
   devices = {"cpu"}
end

rech:add([[
#include "Tensor.h"
namespace xt {
]])
rec:add([[
#include "TensorTH.h"
#include <cmath>
#include "THMath.h"
#include <cstdlib>
#include <array>
#include <cmath>
#include <iostream>
#include "dispatch.h"
#include "TH.h"
#ifdef XT_HAS_CUDA
#include "THC.h"
#endif
#undef THTensor
namespace xt {
static THCState* thcstate()
{
  return defaultContext.thcstate().get();
}
]])

rec:add[[
template<typename T> bool bool_equal_real_real(T a, T b){
  return a == b;
}
]]

local function makedyn(typename, acc)
   if acc then
      return "^^" .. typename
   else
      return "^" .. typename
   end
end

local function undyn(t)
   t = t:gsub('^%^+', '')
   return t
end

local reals = {ByteTensor='unsigned char',
               CharTensor='char',
               ShortTensor='short',
               IntTensor='int',
               LongTensor='long',
               FloatTensor='float',
               DoubleTensor='double'}

local accreals = {ByteTensor='long',
                  CharTensor='long',
                  ShortTensor='long',
                  IntTensor='long',
                  LongTensor='long',
                  FloatTensor='double',
                  DoubleTensor='double'}

for _, device in ipairs(devices) do
   for _,Tensor in ipairs({"ByteTensor", "CharTensor",
                           "ShortTensor", "IntTensor", "LongTensor",
                           "FloatTensor", "DoubleTensor"}) do

      local real = makedyn(reals[Tensor])
      local accreal = makedyn(accreals[Tensor], true)
      local IndexTensor = device == "cpu" and "IndexTensor" or "CudaIndexTensor"
      local LongTensor = device == "cpu" and "LongTensor" or "CudaLongTensor"
      local DoubleTensor = device == "cpu" and "DoubleTensor" or "CudaDoubleTensor"
      local ByteTensor = device == "cpu" and "ByteTensor" or "CudaByteTensor"
      if device == "gpu" then
         Tensor = "Cuda" .. Tensor
         Tensor = Tensor:gsub("CudaFloat", "Cuda")
      end
      Tensor = makedyn(Tensor)

      local function cname(name)
         return string.format('TH%s_%s', Tensor, name)
      end

      local function lastdim(argn)
         return function(arg)
            return string.format("TH%s_nDimension(%s%s)", undyn(Tensor), device == "cpu" and "" or "thcstate(), ", arg.args[argn]:carg())
         end
      end

      local function lastdimarray(argn)
         return function(arg)
            return string.format("TH%s_nDimension(%sarg%d_data[0])", Tensor, device == "cpu" and "" or "thcstate(), ", arg.args[argn].i)
         end
      end


      ----- <core functions> -----
      wrap("narrow",
           cname("narrow"),
           {{name=Tensor, default=true, returned=true},
            {name=Tensor},
            {name="index"},
            {name="index"},
            {name="long"}})

      wrap("select",
           cname("select"),
           {{name=Tensor, default=true, returned=true},
            {name=Tensor},
            {name="index"},
            {name="index"}},
           cname("narrow"),
           {{name=Tensor, default=true, returned=true,
             postcall=function(arg, args)
                return
                   string.format(
                      '%s.tovalue();',
                      arg:ccarg()
                   )
             end},
            {name=Tensor, dim=1},
            {name="index"},
            {name="index"},
            {name="index", invisible=true, default=1}})
      wrap("transpose",
           cname("transpose"),
           {{name=Tensor, default=true, returned=true},
            {name=Tensor},
            {name="index", default=0, defgroup=1},
            {name="index", default=1, defgroup=1}})
      wrap("unfold",
           cname("unfold"),
           {{name=Tensor, default=true, returned=true},
            {name=Tensor},
            {name="index"},
            {name="long"},
            {name="long"}})
      wrap("isContiguous",
           cname("isContiguous"),
           {{name=Tensor},
            {name="boolean", creturned=true}})
      if device == "cpu" then
         interface.write(rec, string.gsub(
                            [[
static void THTensor_contiguous__(THTensor *self, THTensor *o)
{
  THTensor_resizeAs(self, o);
  THTensor_copy(self, o);
}
]], 'Tensor', undyn(Tensor)):gsub('real', undyn(real)))
      elseif device == "gpu" then
         interface.write(rec, string.gsub(
                            [[
static void THTensor_contiguous__(THCState* state, THTensor *self, THTensor *o)
{
  THTensor_resizeAs(state, self, o);
  THTensor_copy(state, self, o);
}
]], 'Tensor', undyn(Tensor)):gsub('real', undyn(real)))
      end
      wrap("contiguous",
           cname("contiguous__"),
           {{name=Tensor, default=true, returned=true},
            {name=Tensor}})
      wrap("index",
           cname("indexSelect"),
           {{name=Tensor, default=true, returned=true},
            {name=Tensor},
            {name="index"},
            {name=LongTensor}})
      wrap("indexCopy",
           cname("indexCopy"),
           {{name=Tensor, notconst=true},
            {name="index"},
            {name=LongTensor},
            {name=Tensor}})
      wrap("indexAdd",
           cname("indexAdd"),
           {{name=Tensor, notconst=true},
            {name="index"},
            {name=LongTensor},
            {name=Tensor}})
      wrap("indexFill",
           cname("indexFill"),
           {{name=Tensor, notconst=true},
            {name="index"},
            {name=LongTensor},
            {name=real}})
      wrap("maskedSelect",
           cname("maskedSelect"),
           {{name=Tensor, default=true, returned=true},
            {name=Tensor},
            {name=ByteTensor}})
      wrap("maskedCopy",
           cname("maskedCopy"),
           {{name=Tensor, notconst=true},
            {name=ByteTensor},
            {name=Tensor}})
      wrap("maskedFill",
           cname("maskedFill"),
           {{name=Tensor, notconst=true},
            {name=ByteTensor},
            {name=real}})
      ----- <core functions> -----

      if not Tensor:match('HalfTensor') then
         wrap("zero",
              cname("zero"),
              {{name=Tensor, returned=true}})

         wrap("fill",
              cname("fill"),
              {{name=Tensor, returned=true},
               {name=real}})

         wrap("zeros",
              cname("zeros"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name="LongArg"}})

         wrap("ones",
              cname("ones"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name="LongArg"}})

         wrap("reshape",
              cname("reshape"),
              {{name=Tensor, default=true, returned=true},
               {name=Tensor},
               {name="LongArg"}})

         wrap("gather",
              cname("gather"),
              {{name=Tensor, default=true, returned=true,
                init=function(arg)
                   return table.concat(
                      {
                         arg.__metatable.init(arg),
                         string.format("THLongStorage* %s_size = THLongTensor_newSizeOf(%s);", arg:carg(), arg.args[4]:carg()),
                         string.format("TH%s_resize(%s, %s_size, NULL);", Tensor, arg:carg(), arg:carg()),
                         string.format("THLongStorage_free(%s_size);", arg:carg())
                      }, '\n')
                end
               },
               {name=Tensor},
               {name="index"},
               {name=IndexTensor, noreadadd=true}})

         wrap("scatter",
              cname("scatter"),
              {{name=Tensor, returned=true},
               {name="index"},
               {name=IndexTensor, noreadadd=true},
               {name=Tensor}})

         wrap("scatter",
              cname("scatterFill"),
              {{name=Tensor, returned=true},
               {name="index"},
               {name=IndexTensor, noreadadd=true},
               {name=real}})

         wrap("dot",
              cname("dot"),
              {{name=Tensor},
               {name=Tensor},
               {name=accreal, creturned=true}})

         wrap("equal",
              cname("equal"),
              {{name=Tensor},
               {name=Tensor},
               {name="boolean", creturned=true}})
         if device == "cpu" then
            wrap("equal",
                 "bool_equal_real_real",
                 {{name=real},
                  {name=real},
                  {name="boolean", creturned=true}})
         end

         wrap("add",
              cname("add"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=real}},
              cname("cadd"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=real, default=1},
               {name=Tensor}})

         wrap("csub",
              cname("sub"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=real}},
              cname("csub"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=real, default=1},
               {name=Tensor}})

         wrap("mul",
              cname("mul"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=real}})

         wrap("div",
              cname("div"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=real}})

         wrap("lshift",
              cname("lshift"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=real}})

         wrap("rshift",
              cname("rshift"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=real}})

         wrap("fmod",
              cname("fmod"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=real}})

         wrap("remainder",
              cname("remainder"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=real}})

         wrap("band",
              cname("bitand"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=real}})

         wrap("bor",
              cname("bitor"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=real}})

         wrap("bxor",
              cname("bitxor"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=real}})

         -- mod alias
         wrap("mod",
              cname("fmod"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=real}})

         wrap("clamp",
              cname("clamp"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=real},
               {name=real}})


         if device ~= "gpu" then
            wrap("match",
                 cname("match"),
                 {{name=Tensor, default=true, returned=true, method={default='nil'}},
                  {name=Tensor},
                  {name=Tensor},
                  {name=real, default=1}
                 })
         end
         
         wrap("cmul",
              cname("cmul"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=Tensor}})

         wrap("cpow",
              cname("cpow"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=Tensor}})

         wrap("cdiv",
              cname("cdiv"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=Tensor}})

         wrap("clshift",
              cname("clshift"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=Tensor}})

         wrap("crshift",
              cname("crshift"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=Tensor}})

         wrap("cfmod",
              cname("cfmod"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=Tensor}})

         wrap("cremainder",
              cname("cremainder"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=Tensor}})

         wrap("cband",
              cname("cbitand"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=Tensor}})

         wrap("cbor",
              cname("cbitor"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=Tensor}})

         wrap("cbxor",
              cname("cbitxor"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=Tensor}})

         -- cmod alias
         wrap("cmod",
              cname("cfmod"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=Tensor}})

         wrap("addcmul",
              cname("addcmul"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=real, default=1},
               {name=Tensor},
               {name=Tensor}})

         wrap("addcdiv",
              cname("addcdiv"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=real, default=1},
               {name=Tensor},
               {name=Tensor}})

         wrap("mv",
              cname("addmv"),
              {{name=Tensor, default=true, returned=true, method={default='nil'},
                precall=function(arg, args)
                   return table.concat(
                      {
                         string.format("TH%s_resize1d(%s%s, %s->size[0]);", undyn(Tensor), device == "cpu" and "" or "thcstate(), ", arg:carg(), args[5]:carg()),
                         string.format("TH%s_zero(%s%s);", undyn(Tensor), device == "cpu" and "" or "thcstate(), ", arg:carg()),
                      }, '\n')
                end,
               },
               {name=real, default=0, invisible=true},
               {name=Tensor, default=1, invisible=true},
               {name=real, default=1, invisible=true},
               {name=Tensor, dim=2},
               {name=Tensor, dim=1}}
         )

         wrap("mm",
              cname("addmm"),
              {{name=Tensor, default=true, returned=true, method={default='nil'},
                precall=function(arg, args)
                   return table.concat(
                      {
                         string.format("TH%s_resize2d(%s%s, %s->size[0], %s->size[1]);", undyn(Tensor), device == "cpu" and "" or "thcstate(), ", arg:carg(), args[5]:carg(), args[6]:carg()),
                         string.format("TH%s_zero(%s%s);", undyn(Tensor), device == "cpu" and "" or "thcstate(), ", arg:carg()),
                      }, '\n')
                end,
               },
               {name=real, default=0, invisible=true},
               {name=Tensor, default=1, invisible=true},
               {name=real, default=1, invisible=true},
               {name=Tensor, dim=2},
               {name=Tensor, dim=2}}
         )

         wrap("bmm",
              cname("baddbmm"),
              {{name=Tensor, default=true, returned=true, method={default='nil'},
                precall=function(arg, args)
                   return table.concat(
                      {
                         string.format("TH%s_resize3d(%s%s, %s->size[0], %s->size[1], %s->size[2]);",
                                       undyn(Tensor), device == "cpu" and "" or "thcstate(), ", arg:carg(), args[5]:carg(), args[5]:carg(), args[6]:carg()),
                         string.format("TH%s_zero(%s%s);", undyn(Tensor), device == "cpu" and "" or "thcstate(), ", arg:carg()),
                      }, '\n')
                end,
               },
               {name=real, default=0, invisible=true},
               {name=Tensor, default=1, invisible=true},
               {name=real, default=1, invisible=true},
               {name=Tensor, dim=3},
               {name=Tensor, dim=3}}
         )

         wrap("ger",
              cname("addr"),
              {{name=Tensor, default=true, returned=true, method={default='nil'},
                precall=function(arg, args)
                   return table.concat(
                      {
                         string.format("TH%s_resize2d(%s%s, %s->size[0], %s->size[0]);", undyn(Tensor), device == "cpu" and "" or "thcstate(), ", arg:carg(), args[5]:carg(), args[6]:carg()),
                         string.format("TH%s_zero(%s%s);", undyn(Tensor), device == "cpu" and "" or "thcstate(), ", arg:carg()),
                      }, '\n')
                end
               },
               {name=real, default=1, invisible=true},
               {name=Tensor, default=1, invisible=true},
               {name=real, default=1, invisible=true},
               {name=Tensor, dim=1},
               {name=Tensor, dim=1}}
         )

         for _,f in ipairs({
                              {name="addmv",   dim1=1, dim2=2, dim3=1},
                              {name="addmm",   dim1=2, dim2=2, dim3=2},
                              {name="addr",    dim1=2, dim2=1, dim3=1},
                              {name="addbmm",  dim1=2, dim2=3, dim3=3},
                              {name="baddbmm", dim1=3, dim2=3, dim3=3},
                           }
                          ) do

            wrap(f.name,
                 cname(f.name),
                 {{name=Tensor, default=true, returned=true},
                  {name=real, default=1},
                  {name=Tensor, dim=f.dim1},
                  {name=real, default=1},
                  {name=Tensor, dim=f.dim2},
                  {name=Tensor, dim=f.dim3}})
         end

         wrap("numel",
              cname("numel"),
              {{name=Tensor},
               {name="long", creturned=true}})

         for _,name in ipairs({"cumsum", "cumprod"}) do
            wrap(name,
                 cname(name),
                 {{name=Tensor, default=true, returned=true},
                  {name=Tensor},
                  {name="index", default=1}})
         end

         wrap("sum",
              cname("sumall"),
              {{name=Tensor},
               {name=accreal, creturned=true}},
              cname("sum"),
              {{name=Tensor, default=true, returned=true},
               {name=Tensor},
               {name="index"},
               {name="boolean", default=true, invisible=true}})

         wrap("prod",
              cname("prodall"),
              {{name=Tensor},
               {name=accreal, creturned=true}},
              cname("prod"),
              {{name=Tensor, default=true, returned=true},
               {name=Tensor},
               {name="index"},
               {name="boolean", default=true, invisible=true}})

         for _,name in ipairs({"min", "max"}) do
            wrap(name,
                 cname(name .. "all"),
                 {{name=Tensor},
                  {name=real, creturned=true}},
                 cname(name),
                 {{name=Tensor, default=true, returned=true, defgroup=1},
                  {name=IndexTensor, default=true, returned=true, noreadadd=true, defgroup=1},
                  {name=Tensor},
                  {name="index"},
                  {name="boolean", default=true, invisible=true}})
         end

         for _,name in ipairs({"cmin", "cmax"}) do
            wrap(name,
                 cname(name),
                 {{name=Tensor, default=true, returned=true},
                  {name=Tensor, method={default=1}},
                  {name=Tensor}},
                 cname(name .. "Value"),
                 {{name=Tensor, default=true, returned=true},
                  {name=Tensor, method={default=1}},
                  {name=real}})
         end

         wrap("trace",
              cname("trace"),
              {{name=Tensor},
               {name=accreal, creturned=true}})

         wrap("cross",
              cname("cross"),
              {{name=Tensor, default=true, returned=true},
               {name=Tensor},
               {name=Tensor},
               {name="index", default=0}})

         wrap("diag",
              cname("diag"),
              {{name=Tensor, default=true, returned=true},
               {name=Tensor},
               {name="long", default=0}})

         if device ~= "gpu" then
            wrap("eye",
                 cname("eye"),
                 {{name=Tensor, default=true, returned=true, method={default='nil'}},
                  {name="long"},
                  {name="long", default=0}})
         end

         wrap("range",
              cname("range"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=accreal},
               {name=accreal},
               {name=accreal, default=1}})

         if device ~= "gpu" then
            wrap("randperm",
                 cname("randperm"),
                 {{name=Tensor, default=true, returned=true, method={default='nil'},
                   -- postcall=function(arg)
                   --    return table.concat(
                   --       {
                   --          arg.__metatable.postcall(arg),
                   --          string.format("TH%s_add(%s, %s, 1);", Tensor, arg:carg(), arg:carg())
                   --       }, '\n')
                   -- end
                  },
                  {name="Generator", default=true},
                  {name="long"}})
         end
         
         wrap("sort",
              cname("sort"),
              {{name=Tensor, default=true, returned=true, defgroup=1},
               {name=IndexTensor, default=true, returned=true, noreadadd=true, defgroup=1},
               {name=Tensor},
               {name="index", default=lastdim(3)},
               {name="boolean", default=false, enum={name="TensorOrder", "kAscend", "kDescend"}}}
         )
         wrap("topk",
              cname("topk"),
              {{name=Tensor, default=true, returned=true, defgroup=1},
               {name=IndexTensor, default=true, returned=true, noreadadd=true, defgroup=1},
               {name=Tensor},
               {name="long"},
               {name="index", default=lastdim(3)},
               {name="boolean", default=false, defgroup=2},
               {name="boolean", default=false, defgroup=2}})
         if device ~= "gpu" then
            wrap("kthvalue",
                 cname("kthvalue"),
                 {{name=Tensor, default=true, returned=true, defgroup=1},
                  {name=IndexTensor, default=true, returned=true, noreadadd=true, defgroup=1},
                  {name=Tensor},
                  {name="long"},
                  {name="index", default=lastdim(3)},
                  {name="boolean", default=true, invisible=true}})
         end
         wrap("mode",
              cname("mode"),
              {{name=Tensor, default=true, returned=true, defgroup=1},
               {name=IndexTensor, default=true, returned=true, noreadadd=true, defgroup=1},
               {name=Tensor},
               {name="index", default=lastdim(3)},
               {name="boolean", default=true, invisible=true}})
         if device ~= "gpu" then
            wrap("median",
                 cname("median"),
                 {{name=Tensor, default=true, returned=true, defgroup=1},
                  {name=IndexTensor, default=true, returned=true, noreadadd=true, defgroup=1},
                  {name=Tensor},
                  {name="index", default=lastdim(3)},
                  {name="boolean", default=true, invisible=true}})
         end
         wrap("tril",
              cname("tril"),
              {{name=Tensor, default=true, returned=true},
               {name=Tensor},
               {name="int", default=0}})
         wrap("triu",
              cname("triu"),
              {{name=Tensor, default=true, returned=true},
               {name=Tensor},
               {name="int", default=0}})
         wrap("cat",
              cname("cat"),
              {{name=Tensor, default=true, returned=true},
               {name=Tensor},
               {name=Tensor},
               {name="index", default=-1}})
         --[[ DEBUG: NYI
            cname("catArray"),
            {{name=Tensor, default=true, returned=true},
            {name=Tensor .. "Array"},
            {name="index", default=-1}})
         --]]
         if Tensor:match('ByteTensor') and device ~= "gpu" then -- we declare this only once
            interface.write(rec,
                            [[
static long THRandom_random2__(THGenerator *gen, long a, long b)
{
  THArgCheck(b >= a, 2, "upper bound must be larger than lower bound");
  return((THRandom_random(gen) % (b+1-a)) + a);
}

static long THRandom_random1__(THGenerator *gen, long b)
{
  THArgCheck(b > 0, 1, "upper bound must be strictly positive");
  return(THRandom_random(gen) % b + 1);
}
         ]])
         end

         interface.write(rec, string.gsub(
                            [[
static void THTensor_random2__(THTensor *self, THGenerator *gen, long a, long b)
{
  THArgCheck(b >= a, 2, "upper bound must be larger than lower bound");
  TH_TENSOR_APPLY(real, self, *self_data = ((THRandom_random(gen) % (b+1-a)) + a);)
}

static void THTensor_random1__(THTensor *self, THGenerator *gen, long b)
{
  THArgCheck(b > 0, 1, "upper bound must be strictly positive");
  TH_TENSOR_APPLY(real, self, *self_data = (THRandom_random(gen) % b + 1);)
}
]], 'Tensor', undyn(Tensor)):gsub('real', undyn(real)))

         if Tensor:match('ByteTensor') then
            if device == "cpu" then
               wrap('random',
                    'THRandom_random2__',
                    {{name='Generator', default=true},
                     {name='long'},
                     {name='long'},
                     {name='long', creturned=true}},
                    'THRandom_random1__',
                    {{name='Generator', default=true},
                     {name='long'},
                     {name='long', creturned=true}},
                    'THRandom_random',
                    {{name='Generator', default=true},
                     {name='long', creturned=true}})
               wrap("geometric",
                    "THRandom_geometric",
                    {{name="Generator", default=true},
                     {name="double"},
                     {name="double", creturned=true}})
               wrap("bernoulli",
                    "THRandom_bernoulli",
                    {{name="Generator", default=true},
                     {name="double", default=0.5},
                     {name="double", creturned=true}})
            end
         end

         if device == "cpu" then
            wrap("random",
                 cname("random2__"),
                 {{name=Tensor, returned=true},
                  {name='Generator', default=true},
                  {name='long'},
                  {name='long'}},
                 cname("random1__"),
                 {{name=Tensor, returned=true},
                  {name='Generator', default=true},
                  {name='long'}},
                 cname("random"),
                 {{name=Tensor, returned=true},
                  {name='Generator', default=true}})
         end

         wrap("geometric",
              cname("geometric"),
              {{name=Tensor, returned=true},
               {name="Generator", default=true},
               {name="double"}})

         wrap("bernoulli",
              -- DEBUG: NYI
              -- cname("bernoulli_FloatTensor"),
              -- {{name=Tensor, returned=true},
              --  {name="Generator", default=true},
              --  {name="FloatTensor"}},
              cname("bernoulli_DoubleTensor"),
              {{name=Tensor, returned=true},
               {name="Generator", default=true},
               {name=DoubleTensor}},
              cname("bernoulli"),
              {{name=Tensor, returned=true},
               {name="Generator", default=true},
               {name="double", default=0.5}})


         wrap("squeeze",
              cname("squeeze"),
              {{name=Tensor, default=true, returned=true,
                postcall=function(arg, args)
                   return table.concat{
                      string.format('if(%s->nDimension == 1 && %s->size[0] == 1) { ', arg:carg(), arg:carg()),
                      string.format(
                         '%s.value((%s)TH%s_get1d(%s%s, 0)); }',
                         arg:ccarg(),
                         types.tensorbase(undyn(Tensor)), -- extra cast because of char/long
                         undyn(Tensor),
                         device == "cpu" and "" or "thcstate(), ",
                         arg:carg()),
                      '\n'
                   }
                end},
               {name=Tensor}})
         --[===[



            cname("squeeze1d"),
            {{name=Tensor, default=true, returned=true,

            postcall=
            function(arg)
            local txt = {}
            if arg.returned then
            table.insert(txt, string.format('if(!hasdims && arg%d->nDimension == 1 && arg%d->size[0] == 1)', arg.i, arg.i)) -- number
            table.insert(txt, string.format('lua_pushnumber(L, (lua_Number)(*TH%s_data(arg%d)));}', Tensor, arg.i))
            end
            return table.concat(txt, '\n')
            end},

            {name=Tensor,

            precall=
            function(arg)
            return string.format('{int hasdims = arg%d->nDimension > 1;', arg.i)
            end},

            {name="index"}})
         --]===]
         wrap("sign",
              cname("sign"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}}})

         if device ~= "gpu" then
            wrap("conv2",
                 cname("conv2Dmul"),
                 {{name=Tensor, default=true, returned=true},
                  {name=real, default=0, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name=Tensor, dim=2},
                  {name=Tensor, dim=2},
                  {name=real, default=1, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name='charoption', values={'V', 'F'}, default='V'},
                  {name='charoption', default="C", invisible=true}},
                 cname("conv2Dcmul"),
                 {{name=Tensor, default=true, returned=true},
                  {name=real, default=0, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name=Tensor, dim=3},
                  {name=Tensor, dim=3},
                  {name=real, default=1, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name='charoption', values={'V', 'F'}, default='V'},
                  {name='charoption', default="C", invisible=true}},
                 cname("conv2Dmv"),
                 {{name=Tensor, default=true, returned=true},
                  {name=real, default=0, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name=Tensor, dim=3},
                  {name=Tensor, dim=4},
                  {name=real, default=1, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name='charoption', values={'V', 'F'}, default='V'},
                  {name='charoption', default="C", invisible=true}}
            )

            wrap("xcorr2",
                 cname("conv2Dmul"),
                 {{name=Tensor, default=true, returned=true},
                  {name=real, default=0, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name=Tensor, dim=2},
                  {name=Tensor, dim=2},
                  {name=real, default=1, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name='charoption', values={'V', 'F'}, default='V'},
                  {name='charoption', default="X", invisible=true}},
                 cname("conv2Dcmul"),
                 {{name=Tensor, default=true, returned=true},
                  {name=real, default=0, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name=Tensor, dim=3},
                  {name=Tensor, dim=3},
                  {name=real, default=1, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name='charoption', values={'V', 'F'}, default='V'},
                  {name='charoption', default="X", invisible=true}},
                 cname("conv2Dmv"),
                 {{name=Tensor, default=true, returned=true},
                  {name=real, default=0, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name=Tensor, dim=3},
                  {name=Tensor, dim=4},
                  {name=real, default=1, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name='charoption', values={'V', 'F'}, default='V'},
                  {name='charoption', default="X", invisible=true}}
            )

            wrap("conv3",
                 cname("conv3Dmul"),
                 {{name=Tensor, default=true, returned=true},
                  {name=real, default=0, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name=Tensor, dim=3},
                  {name=Tensor, dim=3},
                  {name=real, default=1, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name='charoption', values={'V', 'F'}, default='V'},
                  {name='charoption', default="C", invisible=true}},
                 cname("conv3Dcmul"),
                 {{name=Tensor, default=true, returned=true},
                  {name=real, default=0, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name=Tensor, dim=4},
                  {name=Tensor, dim=4},
                  {name=real, default=1, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name='charoption', values={'V', 'F'}, default='V'},
                  {name='charoption', default="C", invisible=true}},
                 cname("conv3Dmv"),
                 {{name=Tensor, default=true, returned=true},
                  {name=real, default=0, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name=Tensor, dim=4},
                  {name=Tensor, dim=5},
                  {name=real, default=1, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name='charoption', values={'V', 'F'}, default='V'},
                  {name='charoption', default="C", invisible=true}}
            )

            wrap("xcorr3",
                 cname("conv3Dmul"),
                 {{name=Tensor, default=true, returned=true},
                  {name=real, default=0, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name=Tensor, dim=3},
                  {name=Tensor, dim=3},
                  {name=real, default=1, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name='charoption', values={'V', 'F'}, default='V'},
                  {name='charoption', default="X", invisible=true}},
                 cname("conv3Dcmul"),
                 {{name=Tensor, default=true, returned=true},
                  {name=real, default=0, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name=Tensor, dim=4},
                  {name=Tensor, dim=4},
                  {name=real, default=1, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name='charoption', values={'V', 'F'}, default='V'},
                  {name='charoption', default="X", invisible=true}},
                 cname("conv3Dmv"),
                 {{name=Tensor, default=true, returned=true},
                  {name=real, default=0, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name=Tensor, dim=4},
                  {name=Tensor, dim=5},
                  {name=real, default=1, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name=real, default=1, invisible=true},
                  {name='charoption', values={'V', 'F'}, default='V'},
                  {name='charoption', default="X", invisible=true}}
            )
         end -- device ~= "gpu"

         for _,name in pairs({'lt','gt','le','ge','eq','ne'}) do
            wrap(name,
                 cname(name .. 'Value'),
                 {{name=ByteTensor,default=true, returned=true},
                  {name=Tensor},
                  {name=real}},
                 -- cname(name .. 'ValueT'), DEBUG: NYI
                 -- {{name=Tensor, returned=true},
                 --  {name=Tensor},
                 --  {name=real}},
                 cname(name .. 'Tensor'),
                 {{name=ByteTensor,default=true, returned=true},
                  {name=Tensor},
                  {name=Tensor}})
            -- cname(name .. 'TensorT'), DEBUG: NYI
            -- {{name=Tensor, returned=true},
            --  {name=Tensor},
            --  {name=Tensor}})
         end

         wrap("nonzero",
              cname("nonzero"),
              {{name=IndexTensor, default=true, returned=true},
               {name=Tensor}})

      end  -- ~= HalfTensor

      if Tensor:match('ByteTensor') then
         -- Logical accumulators only apply to ByteTensor
         for _,name in ipairs({'all', 'any'}) do
            wrap(name,
                 cname('logical' .. name),
                 {{name=Tensor},
                  {name="boolean", creturned=true}})
         end
      end

      if Tensor:match('IntTensor') then
         wrap("abs",
              cname("abs"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}}})
         if device == "cpu" then
            wrap("abs",
                 "std::abs",
                 {{name=real},
                  {name=real, creturned=true}})
         end
      elseif Tensor:match('LongTensor') then
         wrap("abs",
              cname("abs"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}}})
         if device == "cpu" then
            wrap("abs",            
                 "std::labs",
                 {{name=real},
                  {name=real, creturned=true}})
         end
      end

      if Tensor:match('FloatTensor') or Tensor:match('DoubleTensor') or Tensor:match('CudaTensor') then

         wrap("mean",
              cname("meanall"),
              {{name=Tensor},
               {name=accreal, creturned=true}},
              cname("mean"),
              {{name=Tensor, default=true, returned=true},
               {name=Tensor},
               {name="index"},
               {name="boolean", default=true, invisible=true}})

         for _,name in ipairs({"var", "std"}) do
            wrap(name,
                 cname(name .. "all"),
                 {{name=Tensor},
                  {name=accreal, creturned=true}},
                 cname(name),
                 {{name=Tensor, default=true, returned=true},
                  {name=Tensor},
                  {name="index"},
                  {name="boolean", default=false},
                  {name="boolean", default=true, invisible=true}})
         end

         if device ~= "gpu" then
            wrap("histc",
                 cname("histc"),
                 {{name=Tensor, default=true, returned=true},
                  {name=Tensor},
                  {name="long",default=100},
                  {name="double",default=0, invisible=true},
                  {name="double",default=0, invisible=true}},
                 cname("histc"),
                 {{name=Tensor, default=true, returned=true},
                  {name=Tensor},
                  {name="long"},
                  {name="double"},
                  {name="double"}}
            )
            wrap("bhistc",
                 cname("bhistc"),
                 {{name=Tensor, default=true, returned=true},
                  {name=Tensor},
                  {name="long",default=100},
                  {name="double",default=0, invisible=true},
                  {name="double",default=0, invisible=true}},
                 cname("bhistc"),
                 {{name=Tensor, default=true, returned=true},
                  {name=Tensor},
                  {name="long"},
                  {name="double"},
                  {name="double"}})
         end

         wrap("norm",
              cname("normall"),
              {{name=Tensor},
               {name=real, default=2},
               {name=accreal, creturned=true}},
              cname("norm"),
              {{name=Tensor, default=true, returned=true},
               {name=Tensor},
               {name=real},
               {name="index"},
               {name="boolean", default=true, invisible=true}})

         wrap("renorm",
              cname("renorm"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=real},
               {name="index"},
               {name=real}})

         wrap("dist",
              cname("dist"),
              {{name=Tensor},
               {name=Tensor},
               {name=real, default=2},
               {name=accreal, creturned=true}})

         wrap("linspace",
              cname("linspace"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=real},
               {name=real},
               {name="long", default=100}})

         wrap("logspace",
              cname("logspace"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=real},
               {name=real},
               {name="long", default=100}})

         for _,name in ipairs({"log", "log1p", "exp",
                               "cos", "acos", "cosh",
                               "sin", "asin", "sinh",
                               "tan", "atan", "tanh",
                               "sqrt", "round", "ceil",
                               "floor", "trunc", }) do
            wrap(name,
                 cname(name),
                 {{name=Tensor, default=true, returned=true, method={default='nil'}},
                  {name=Tensor, method={default=1}}})
            if device == "cpu" then
               wrap(name,
                    "std::" .. name,
                    {{name=real},
                     {name=real, creturned=true}})
            end
         end

         wrap("abs",
              cname("abs"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}}})
         if device == "cpu" then
            wrap("abs",
                 "fabs",
                 {{name=real},
                  {name=real, creturned=true}})
         end

         wrap("frac",
              cname("frac"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}}})
         if device == "cpu" then
            wrap("frac",
                 "TH_frac",
                 {{name=real},
                  {name=real, creturned=true}})
         end

         wrap("rsqrt",
              cname("rsqrt"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}}})
         if device == "cpu" then
            wrap("rsqrt",
                 "TH_rsqrt",
                 {{name=real},
                  {name=real, creturned=true}})
         end

         wrap("sigmoid",
              cname("sigmoid"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}}})
         if device == "cpu" then
            wrap("sigmoid",
                 "TH_sigmoid",
                 {{name=real},
                  {name=real, creturned=true}})
         end

         wrap("neg",
              cname("neg"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}}})

         wrap("cinv",
              cname("cinv"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}}})

         wrap("lerp",
              cname("lerp"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=Tensor},
               {name=real}})
         if device == "cpu" then
            wrap("lerp",
                 "TH_lerp",
                 {{name=real},
                  {name=real},
                  {name=real},
                  {name=real, creturned=true}})
         end

         if not (device == "gpu" and Tensor:match('Double')) then
            wrap("atan2",
                 cname("atan2"),
                 {{name=Tensor, default=true, returned=true, method={default='nil'}},
                  {name=Tensor, method={default=1}},
                  {name=Tensor}})
         end
         if device == "cpu" then
            wrap("atan2",
                 "std::atan2",
                 {{name=real},
                  {name=real},
                  {name=real, creturned=true}})
         end

         wrap("pow",
              cname("pow"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}},
               {name=real}},
              cname("tpow"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=real},
               {name=Tensor, method={default=1}}})
         if device == "cpu" then
            wrap("pow",           
                 "std::pow",
                 {{name=real},
                  {name=real},
                  {name=real, creturned=true}})
         end

         wrap("rand",
              cname("rand"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name='Generator', default=true},
               {name="LongArg"}})

         wrap("randn",
              cname("randn"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name='Generator', default=true},
               {name="LongArg"}})

         wrap("multinomial",
              cname("multinomial"),
              {{name=IndexTensor, default=true, returned=true, method={default='nil'}},
               {name='Generator', default=true},
               {name=Tensor},
               {name="int"},
               {name="boolean", default=false}})

         for _,f in ipairs({{name='uniform', a=0, b=1},
                            {name='normal', a=0, b=1},
                            {name='cauchy', a=0, b=1},
                            {name='logNormal', a=1, b=2}}) do
            if Tensor:match('DoubleTensor') and device == "cpu" then
               wrap(f.name,
                    string.format("THRandom_%s", f.name),
                    {{name='Generator', default=true},
                     {name="double", default=f.a, defgroup=1},
                     {name="double", default=f.b, defgroup=1},
                     {name="double", creturned=true}})
            end
            wrap(f.name,
                 cname(f.name),
                 {{name=Tensor, returned=true},
                  {name='Generator', default=true},
                  {name=real, default=f.a, defgroup=1},
                  {name=real, default=f.b, defgroup=1}})
         end

         for _,f in ipairs({{name='exponential'}}) do

            if Tensor:match('DoubleTensor') and device == "cpu" then
               wrap(f.name,
                    string.format("THRandom_%s", f.name),
                    {{name='Generator', default=true},
                     {name="double", default=f.a},
                     {name="double", creturned=true}})
            end
            wrap(f.name,
                 cname(f.name),
                 {{name=Tensor, returned=true},
                  {name='Generator', default=true},
                  {name=real, default=f.a}})
         end

         for _,name in ipairs({"gesv","gels"}) do
            wrap(name,
                 cname(name),
                 {{name=Tensor, returned=true, default=true, defgroup=1},
                  {name=Tensor, returned=true, default=true, defgroup=1},
                  {name=Tensor},
                  {name=Tensor}}
            )
         end

         if device ~= "gpu" then
            wrap("trtrs",
                 cname("trtrs"),
                 {{name=Tensor, returned=true, default=true, defgroup=1},
                  {name=Tensor, returned=true, default=true, defgroup=1},
                  {name=Tensor},
                  {name=Tensor},
                  {name='charoption', values={'U', 'L'}, default='U', defgroup=2},  -- uplo
                  {name='charoption', values={'N', 'T'}, default='N', defgroup=2},  -- trans
                  {name='charoption', values={'N', 'U'}, default='N', defgroup=2}}  -- diag
            )
         end

         wrap("symeig",
              cname("syev"),
              {{name=Tensor, returned=true, default=true, defgroup=1},
               {name=Tensor, returned=true, default=true, defgroup=1},
               {name=Tensor},
               {name='charoption', values={'N', 'V'}, default='N', defgroup=2},
               {name='charoption', values={'U', 'L'}, default='U', defgroup=2}}
         )

         wrap("eig",
              cname("geev"),
              {{name=Tensor, returned=true, default=true, defgroup=1},
               {name=Tensor, returned=true, default=true, defgroup=1},
               {name=Tensor},
               {name='charoption', values={'N', 'V'}, default='N'}}
         )

         wrap("svd",
              cname("gesvd"),
              {{name=Tensor, returned=true, default=true, defgroup=1},
               {name=Tensor, returned=true, default=true, defgroup=1},
               {name=Tensor, returned=true, default=true, defgroup=1},
               {name=Tensor},
               {name='charoption', values={'A', 'S'}, default='S'}}
         )

         wrap("inverse",
              cname("getri"),
              {{name=Tensor, returned=true, default=true},
               {name=Tensor}}
         )

         wrap("potrf",
              cname("potrf"),
              {{name=Tensor, returned=true, default=true},
               {name=Tensor},
               {name='charoption', values={'U', 'L'}, default='U'}} -- uplo
         )

         wrap("potrs",
              cname("potrs"),
              {{name=Tensor, returned=true, default=true},
               {name=Tensor},
               {name=Tensor},
               {name='charoption', values={'U', 'L'}, default='U'}} -- uplo
         )
         wrap("potri",
              cname("potri"),
              {{name=Tensor, returned=true, default=true},
               {name=Tensor},
               {name='charoption', values={'U', 'L'}, default='U'}} -- uplo
         )
         if device ~= "gpu" then
            wrap("pstrf",
                 cname("pstrf"),
                 {{name=Tensor, returned=true, default=true, defgroup=1},
                  {name='IntTensor', returned=true, default=true, defgroup=1},
                  {name=Tensor},
                  {name='charoption', values={'U', 'L'}, default='U'},  -- uplo
                  {name=real, invisible=true, default=-1}} -- wtf?
            )
         end
         wrap("qr",
              cname("qr"),
              {{name=Tensor, returned=true, default=true, defgroup=1},
               {name=Tensor, returned=true, default=true, defgroup=1},
               {name=Tensor}}
         )
         if device ~= "gpu" then
            wrap("geqrf",
                 cname("geqrf"),
                 {{name=Tensor, returned=true, default=true, defgroup=1},
                  {name=Tensor, returned=true, default=true, defgroup=1},
                  {name=Tensor}}
            )
            wrap("orgqr",
                 cname("orgqr"),
                 {{name=Tensor, returned=true, default=true},
                  {name=Tensor},
                  {name=Tensor}}
            )
            wrap("ormqr",
                 cname("ormqr"),
                 {{name=Tensor, returned=true, default=true},
                  {name=Tensor},
                  {name=Tensor},
                  {name=Tensor},
                  {name='charoption', values={'L', 'R'}, default='L', defgroup=1},
                  {name='charoption', values={'N', 'T'}, default='N', defgroup=1}}
            )
         end
      end
   end
end

interface.implement(rech, rec)

------ copy ------
rec:add([[
static std::array< std::array<std::function<void (Tensor&, const Tensor&)>, 14>, 14> copy_mt = {{
]])

rec:indent()
for dd,ddv in ipairs(devices) do
   for d,dt in ipairs(types.TensorTypes) do
      local txt = {}
      rec:add("{{")
      rec:indent()
      local DST = (ddv == "cpu" and "" or "Cuda") .. dt.name
      DST = DST:gsub("CudaFloat", "Cuda")
      for ss,ssv in ipairs(devices) do
         for s,st in ipairs(types.TensorTypes) do
            local SRC = (ssv == "cpu" and "" or "Cuda") .. st.name
            local SRX = SRC -- !!!
            SRC = SRC:gsub("CudaFloat", "Cuda")
            local z = "[](Tensor& d, const Tensor& s) { THDSTTensor_copySRX(STATEd.THTensor<THDSTTensor>(), s.THTensor<THSRCTensor>()); }"
            z = z:gsub("SRC", SRC):gsub("DST", DST):gsub("SRX", SRX):gsub("STATE", (ddv=="gpu" or ssv=="gpu") and "thcstate(), " or "")
            table.insert(txt, z)
         end
      end
      rec:add(table.concat(txt, ",\n"))
      rec:unindent()
      rec:add(string.format("}}%s", ((d==#dt and dd==#ddv) and "" or ",")))
   end
end
rec:unindent()

rec:add([[
}};
]])
rech:add([[
void copy_(Tensor& d, const Tensor& s);
]])
rec:add([[
void copy_(Tensor& d, const Tensor& s)
{
  int64_t didx = d.type();
  int64_t sidx = s.type();
  TensorDevice ddev = d.device();
  TensorDevice sdev = s.device();
  if(ddev == kGPU) {
    didx += 7;
  } else if(ddev != kCPU) {
    throw std::invalid_argument("unsupported destination device");
  }
  if(sdev == kGPU) {
    sidx += 7;
  } else if(sdev != kCPU) {
    throw std::invalid_argument("unsupported source device");
  }
   copy_mt.at(didx).at(sidx)(d, s);
}
]])
------ copy ------

rech:add([[
} // namespace xt
]])
rec:add([[
} // namespace xt
]])

local f = io.open(arg[1], "w")
f:write(rech:tostring())
f:close()

local f = io.open(arg[2], "w")
f:write(rec:tostring())
f:close()

local Record = {}

function Record.new()
   local self = {indent_=0, txt_={}}
   setmetatable(self, {__index=Record})
   return self
end

function Record:indent()
   self.indent_ = self.indent_ + 2
end

function Record:unindent()
   self.indent_ = self.indent_ - 2
end

function Record:add(stuff)
   if stuff then
      stuff = stuff:gsub("\n", "\n" .. string.rep(" ", self.indent_))
      stuff = string.rep(" ", self.indent_) .. stuff
      stuff = stuff:gsub("%s+$", "")
      table.insert(self.txt_, stuff)
   end
end

function Record:tostring()
   return table.concat(self.txt_, "\n")
end

return Record.new

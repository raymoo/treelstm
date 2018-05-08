--[[

  A Tree-LSTM that uses an LSTM to combine inputs

--]]

local InnerLSTMTreeLSTM, parent = torch.class('treelstm.InnerLSTMTreeLSTM', 'treelstm.TreeLSTM')

function InnerLSTMTreeLSTM:__init(config)
  parent.__init(self, config)
  self.gat_output = config.gate_output
  if self.gate_output == nil then self.gate_output = true end

  self.criterion = config.criterion

  self.composer = self:new_composer()
  self.composers = {}

  self.output_module = self:new_output_module()
  self.output_modules = {}
end

-- Input: {in_dim, child_count (table) x 1 x mem_dim}
-- Output: 1 x mem_dim
function InnerLSTMTreeLSTM:new_composer()
  local input = nn.View(1, 1, self.in_dim)() -- 1 x 1 x in_dim
  local child_h = nn.Unsqueeze(2)(nn.JoinTable(1)()) -- child_count x 1 x mem_dim

  -- child_count x 1 x (in_dim + mem_dim) -> child_count x 1 x mem_dim
  local seqblstm = nn.SeqLSTM(self.in_dim + self.mem_dim, self.mem_dim, self.mem_dim)

  local h = nn.Squeeze(2)(seqblstm(nn.JoinTable(3)({input, child_h})))
  
  local composer = nn.gModule({input, child_h}, {h});
  if self.composer ~= nil then
    share_params(composer, self.composer)
  end

  return composer
end

-- Input: mem_dim
-- Output: out_dim
function InnerLSTMTreeLSTM:new_output_module(output_fn)
  if output_fn == nil then return nil end
  local output_module = output_fn()
  if self.output_module ~= nil then
    share_params(output_module, self.output_module)
  end
  return output_module
end

function InnerLSTMTreeLSTM:forward(tree, inputs)
  local loss = 0
  local child_h = {}
  for i = 1, tree.num_children do
    local one_h, child_loss = self:forward(tree.children[i], inputs)
    child_h[i] = one_h
    loss = loss + child_loss
  end

  self:allocate_module(tree, 'composer')
  local h = unpack(tree.composer:forward({inputs[tree.idx], child_h}))
  
  if self.output_module ~= nil then
    self:allocate_module(tree, 'output_module')
    tree.output = tree.output_module:forward(h)
    if self.train and tree.gold_label ~= nil then
      loss = loss + self.criterion:forward(output, tree.gold_label)
    end
  end
  return h, loss
end


# 8-LSTM长短时记忆网络的Python实现代码
def y_list_is(self, y_list, loss_layer):
    """
    Updates diffs by setting target sequence
    with corresponding loss layer.
    Will *NOT* update parameters.  To update parameters,
    call self.lstm_param.apply_diff()
    """
    assert len(y_list) == len(self.x_list)
    idx = len(self.x_list) - 1
    # first node only gets diffs from label ...
    loss = loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
    diff_h = loss_layer.bottom_diff(
        self.lstm_node_list[idx].state.h, y_list[idx])
    # here s is not affecting loss due to h(t+1), hence we set equal to zero
    diff_s = np.zeros(self.lstm_param.mem_cell_ct)
    self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
    idx -= 1

    # ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
    # we also propagate error along constant error carousel using diff_s
    while idx >= 0:
        loss += loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
        diff_h = loss_layer.bottom_diff(
            self.lstm_node_list[idx].state.h, y_list[idx])
        diff_h += self.lstm_node_list[idx + 1].state.bottom_diff_h
        diff_s = self.lstm_node_list[idx + 1].state.bottom_diff_s
        self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
        idx -= 1

    return loss


def bottom_diff(self, pred, label):
    diff = np.zeros_like(pred)
    diff[0] = 2 * (pred[0] - label)
    return diff


def top_diff_is(self, top_diff_h, top_diff_s):
    # notice that top_diff_s is carried along the constant error carousel
    ds = self.state.o * top_diff_h + top_diff_s
    do = self.state.s * top_diff_h
    di = self.state.g * ds
    dg = self.state.i * ds
    df = self.s_prev * ds

    # diffs w.r.t. vector inside sigma / tanh function
    di_input = (1. - self.state.i) * self.state.i * di  # sigmoid diff
    df_input = (1. - self.state.f) * self.state.f * df
    do_input = (1. - self.state.o) * self.state.o * do
    dg_input = (1. - self.state.g ** 2) * dg  # tanh diff

    # diffs w.r.t. inputs
    self.param.wi_diff += np.outer(di_input, self.xc)
    self.param.wf_diff += np.outer(df_input, self.xc)
    self.param.wo_diff += np.outer(do_input, self.xc)
    self.param.wg_diff += np.outer(dg_input, self.xc)
    self.param.bi_diff += di_input
    self.param.bf_diff += df_input
    self.param.bo_diff += do_input
    self.param.bg_diff += dg_input

    # compute bottom diff
    dxc = np.zeros_like(self.xc)
    dxc += np.dot(self.param.wi.T, di_input)
    dxc += np.dot(self.param.wf.T, df_input)
    dxc += np.dot(self.param.wo.T, do_input)
    dxc += np.dot(self.param.wg.T, dg_input)

    # save bottom diffs
    self.state.bottom_diff_s = ds * self.state.f
    self.state.bottom_diff_x = dxc[:self.param.x_dim]
    self.state.bottom_diff_h = dxc[self.param.x_dim:]


wi_diff += np.outer((1.-i)*i*di, xc)
wf_diff += np.outer((1.-i)*i*df, xc)
wo_diff += np.outer((1.-i)*i*do, xc)
wg_diff += np.outer((1.-i)*i*dg, xc)

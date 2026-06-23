from neko_2024_NGNW.nets_v6.abstract.agtfactory import abstract_agent_factory_v6
from neko_2024_NGNW.common.agent_pack.branches_mk3.branch_mk3_stability import \
    (neko_branch_mk3_single_head_no_delay_nobw,neko_branch_mk3_single_head_no_delay,neko_branch_mk3_single_head_ohem01_no_delay_nobw,neko_branch_mk3_single_head_ohem01E_no_delay_nobw)
from neko_2024_NGNW.common.agent_pack.human_filter_factory import neko_rule_based_base_human_filter

from neko_2024_NGNW.common.agent_pack.global_tasks.base_mk3 import neko_base_global_tasks_mk3_log_exp
class pgroute_only_agent_factory_v6_no_delay_no_bw(abstract_agent_factory_v6):
    def set_branch_factory(this):
        this.branch_factory=neko_branch_mk3_single_head_no_delay_nobw();
    def set_global_task_factory(this):
        this.global_task_factory=neko_base_global_tasks_mk3_log_exp();
class rule_based_agent_factory_v6_no_delay_no_bw(abstract_agent_factory_v6):
    def set_branch_factory(this):
        this.branch_factory=neko_branch_mk3_single_head_no_delay_nobw();
    def set_hfc_factory(this):
        this.hfc_factory=neko_rule_based_base_human_filter();
class rule_based_agent_factory_v6_no_delay_no_bw_ohem01E(abstract_agent_factory_v6):
    def set_branch_factory(this):
        this.branch_factory=neko_branch_mk3_single_head_ohem01E_no_delay_nobw();
    def set_hfc_factory(this):
        this.hfc_factory=neko_rule_based_base_human_filter();
class aroute_only_agent_factory_v6_no_delay_no_bw(abstract_agent_factory_v6):
    def set_branch_factory(this):
        this.branch_factory=neko_branch_mk3_single_head_no_delay_nobw();

class aroute_only_agent_factory_v6_no_delay_no_bw_ohem01(abstract_agent_factory_v6):
    def set_branch_factory(this):
        this.branch_factory=neko_branch_mk3_single_head_ohem01_no_delay_nobw();

class aroute_only_agent_factory_v6_no_delay_no_bw_ohem01E(abstract_agent_factory_v6):

    def set_branch_factory(this):
        this.branch_factory=neko_branch_mk3_single_head_ohem01E_no_delay_nobw();
class pgroute_only_agent_factory_v6_no_delay_no_bw_ohem01E(abstract_agent_factory_v6):
    def set_branch_factory(this):
        this.branch_factory=neko_branch_mk3_single_head_ohem01E_no_delay_nobw();
    def set_global_task_factory(this):
        this.global_task_factory=neko_base_global_tasks_mk3_log_exp();
class aroute_only_agent_factory_v6_no_delay(abstract_agent_factory_v6):
    def set_branch_factory(this):
        this.branch_factory=neko_branch_mk3_single_head_no_delay();


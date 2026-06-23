from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_async_agent
from neko_sdk.neko_framework_NG.agents.neko_optim_agent import get_neko_optim_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


# A trainer agent include all training routines and some testing routiones.
# Also holds stuffs to carry ALL these routines
#   neko_modulars(weights and optimizers).
#   bogo modulars (simply something callable).
#   other agents
#   raw dataloaders
#   data augmentation and batching facilities
#   data/module distribution system
# routines and testers need to do their own logging.
class neko_trainer_agent(neko_abstract_async_agent):
    enviroment: neko_environment
    PARAM_devices="devices";
    # a routine will handles its own data queues and backward pass.
    PARAM_routine_dict="routine_dict";
    PARAM_routine_names="routine_names";
    # an optim updates params by its gradients.
    # If you want something smarter than an GD, set your own optim agents here.
    PARAM_optim_names="optim_names";
    PARAM_optim_dict="optim_dict";
    # pretest_dict: agents you want to run before running a test agent in the testlist. (so its dict of dict and dict of list)
    PARAM_pretest_dict="pretest_dict";
    PARAM_pretest_names="pretest_names";
    # posttest_dict: agents you want to run before running a test agent in the testlist. (so its dict of dict and dict of list)
    PARAM_posttest_dict="posttest_dict";
    PARAM_posttest_names="posttest_names";

    PARAM_tester_names="tester_names";
    PARAM_tester_dict="tester_dict";

    PARAM_trainable_tags="trainable_tags";

    PARAM_iter_logger_names="iter_logger_names";
    PARAM_iter_logger_dict="iter_logger_dict";


    def build_agent(this,adict):
        rad={}
        for k in adict:
            rad[k]=adict[k]["agent"](adict[k]["params"]);
        return rad;

    def setup(this,param):
        this.mod_tags=neko_get_arg(this.PARAM_trainable_tags,param,"NEP_skipped_NEP");
        this.devices=neko_get_arg(this.PARAM_devices,param,["cuda:0"]);
        this.routine_dict=this.build_agent(param[this.PARAM_routine_dict]);
        this.routine_names=param[this.PARAM_routine_names];

        this.optim_dict=this.build_agent(neko_get_arg(this.PARAM_optim_dict,param,{"main_optim":get_neko_optim_agent()}));
        this.optim_names=neko_get_arg(this,param,["main_optim"])

        this.pre_test_dict=this.build_agent(param[this.PARAM_pretest_dict]);
        this.pre_test_names=param[this.PARAM_pretest_names];

        this.post_test_names=this.build_agent(param[this.PARAM_posttest_dict]);
        this.post_test_dict = param[this.PARAM_posttest_names];

        this.tester_names = param[this.PARAM_tester_names];
        this.tester_dict=this.build_agent(param[this.PARAM_tester_dict]);

        this.iter_logger_names=param[this.PARAM_iter_logger_names];
        this.iter_logger_dict = this.build_agent(param[this.PARAM_iter_logger_dict]);

        this.epoch_logger_names=param["epoch_logger_names"];
        this.epoch_logger_dict = this.build_agent(param["epoch_logger_dict"]);

        this.check_each=neko_get_arg("check_each",param,20000);


        this.set_private_env=neko_get_arg("set_private_env",param,False);
        this.epoch_cnt=neko_get_arg("epoch_cnt",param);
        this.iter_cnt=neko_get_arg("iter_cnt",param);



    def mount_environment(this,param,environment:neko_environment):
        this.environment=environment;
        for k in this.routine_dict:
            if(k in param):
                this.routine_dict[k]["environment"]=\
                    this.make_private_enviroment(environment,param[k]["remap"]);



    ## Switchs into training or testing mode.
    ## There will be cache makings.

    def take_action(this,_,environment:neko_environment):
        ws=neko_workspace(device=this.devices[0]);
        ws.batch_idx=environment.batch_idx;
        ws.epoch_idx=environment.epoch_idx;
        for k in this.routine_names:
            this.routine_dict[k].take_action(ws,environment);
            # print(k,"done fpbp")
        for k in this.optim_names:
            this.optim_dict[k].take_action(ws,environment);
        for k in this.iter_logger_names:
            this.iter_logger_dict[k].take_action(ws,environment);
        environment.batch_idx += 1;
        # print(ws.logdict);
        pass;
    def dump(this,path):
        for k in this.tester_dict:
            this.tester_dict[k].dump_proto(None, this.environment,path);
        pass;

    def check(this,bs=160):
        for k in this.tester_dict:
            if(k in this.pre_test_names):
                for pk in this.pre_test_names[k]:
                    this.pre_test_dict[k][pk].take_action(None, this.environment);
                    # since agents are stateless, we are not building an agent pool here,
                    # if you do want to share, put them in a module and have the agents associated to call the module.

            rd=this.tester_dict[k].take_action_multibatch(None,this.environment,bs)
            if (k in this.post_test_names):
                # what if we have some stats you say? put it into the environment!
                # Specifically, you put whatever you want to store for more than one iter
                   # into neko_environment.assets_dict
                for pk in this.pre_test_names[k]:
                    this.post_test_dict[k][pk].take_action(None, this.environment);
            return rd;


    def dbg_core(this):
        for k in this.tester_dict:
            if(k in this.pre_test_names):
                for pk in this.pre_test_names[k]:
                    this.pre_test_dict[k][pk].take_action(None, this.environment);
                    # since agents are stateless, we are not building an agent pool here,
                    # if you do want to share, put them in a module and have the agents associated to call the module.

            this.tester_dict[k].take_action(None,this.environment)
            if (k in this.post_test_names):
                # what if we have some stats you say? put it into the environment!
                # Specifically, you put whatever you want to store for more than one iter
                   # into neko_environment.assets_dict
                for pk in this.pre_test_names[k]:
                    this.post_test_dict[k][pk].take_action(None, this.environment);

    def eval_mode(this):
        if(this.mod_tags is not None):
            for tag in this.mod_tags:
                this.environment.modset.eval_mode(tag);
        else:
            this.environment.modset.eval_mode();
    def train_mode(this):
        if (this.mod_tags is not None):
            for tag in this.mod_tags:
                this.environment.modset.train_mode(tag);
        else:
            this.environment.modset.train_mode();
    def eval(this,bs=32):
        this.eval_mode();
        rd=this.check(bs);
        this.train_mode();
        return rd;
    def dumpcache(this,path):
        this.eval_mode();
        this.dump(path);
        this.train_mode();
    def dbg(this):
        this.eval_mode();
        this.dbg_core();
        this.train_mode();

    def check_and_save(this):
        this.eval_mode();
        try:
            this.environment.save_mods();
        except:
            this.environment.save_mods();
            print("saving failed, full disk?")
        this.check();
        this.train_mode();

    def action_loop(this):
        # this.check_and_save();
        for i in range(this.epoch_cnt):
            for j in range(this.iter_cnt):
                if (j and j % this.check_each == 0):
                    for k in this.epoch_logger_names:
                        this.epoch_logger_dict[k].take_action({},this.environment);
                    this.check_and_save();
                this.take_action(None, this.environment);
                # this.check_and_save();
            this.environment.epoch_idx += 1;
            this.environment.batch_idx = 0;
            this.environment.modset.update_opt(this.epoch_cnt);
            this.check_and_save();
            print("epoch done");





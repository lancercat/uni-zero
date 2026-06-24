from osocrNG.modular_agents_ocrNG.output_logging_subs.show_batch_agent import neko_vis_agent
def get_show_batch_agent(prefix,tensor_img_name,
                         raw_gt_text_name,
                         tdict_name="",tensor_proto_img_name="",
                         tensor_beacon_name="",tensor_plabel_name=""):

    return{
                "agent":neko_vis_agent,
                "params":{
                    "iocvt_dict":{
                        neko_vis_agent.OUTPUT_raw_image:prefix,
                        neko_vis_agent.INPUT_tensor_img_name: tensor_img_name,
                        neko_vis_agent.INPUT_tensor_proto_img_name:tensor_proto_img_name ,
                        neko_vis_agent.INPUT_tdict_name:tdict_name,
                        neko_vis_agent.INPUT_raw_gt_text_name:raw_gt_text_name,
                        neko_vis_agent.INPUT_tensor_beacon_name:tensor_beacon_name,
                        neko_vis_agent.INPUT_tensor_plabel_name:tensor_plabel_name,
                    },
                    "modcvt_dict":
                        {

                        }

                }
        };

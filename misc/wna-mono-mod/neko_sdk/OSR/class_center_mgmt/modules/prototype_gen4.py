# bogomodules are certain combination to the modules, they do not hold parameters
# instead they use whatever armed to the module set.
# Different to routines, they are statically associated to a certain set of modules for speed up.
# some time you cannot have a container. That's life.
# say u have module a b c d.
# A uses [ac] B uses [ab] C uses [ad]...
# There is no better way than simply put a,b,c,d in a big basket.

from neko_sdk.neko_framework_NG.bogog2_modules.im_to_feat import gen4_object_to_feat_abstract

# starting from framework NG, bogo modules have no authorize to control the stance of its modules.
# however, we will add an API to fetch their members.
# Authorization now goes back to drivers (trainer, tester, or evolver).
# There might be more bookkeeping, but makes apis more straight forward and bug prone.
class vis_prototyper_gen4(gen4_object_to_feat_abstract):

    def forward(this, normprotos):
        raw_protos=this.fe(normprotos);
        if ("drop" in this.modnames):
            raw_protos = this.drop(raw_protos);
        return raw_protos;

class vis_prototyper_gen4_msr(gen4_object_to_feat_abstract):
    def fe(this,clips):
        features,mask,stat = this.backbone(clips)
        A = this.aggr(features);
        # A=torch.ones_like(A);
        out_emb=(A.unsqueeze(2)*features[-1].unsqueeze(1)).sum(-1).sum(-1)/A.unsqueeze(2).sum(-1).sum(-1);
        return out_emb;


from neko_sdk.OSR.classification.agents.pointwise_translation import neko_point_wise_translation
class neko_single_item_translator(neko_point_wise_translation):
    def translate(this,logits,tdict):
        return list(super().translate(logits, tdict));
    # if all samples has just one target, and that target does not have a complex representation, then here we go.
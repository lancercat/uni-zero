import cv2;
import numpy as np;
import torch;
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
#https://stackoverflow.com/questions/43060479/how-to-get-the-font-pixel-height-using-pils-imagefont-class
from neko_sdk.ocr_modules.fontkit.fntmgmt import fntmgmt

# meta mk3 will have more than one proto for each character, hence we will need new prototyping system...
# If I have time this weekend we will switch.
# Hang on.
class meta_mk3:
    PARAM_MASTER="master"; # master[A] -> a.
    PARAM_SERVANTS="servants"; # servants [a]-> {A};
    PARAM_FOES="foes"; # prior, if you want to define looks alike but different characters, put them here.
    PARAM_RELATIONSHIPS="relationships"; # all character involved in a certain character
    PARAM_CHARACTERS="characters";
    PARAM_LABEL_DICT="label_dict";

    # sptokens, include unknown character, are now managed by models.
    # only facts stored here :)
    def init_meta(this,characters):
        meta={};
        meta[this.PARAM_MASTER] = {};
        meta[this.PARAM_SERVANTS] = {};
        meta[this.PARAM_FOES] = {};
        meta[this.PARAM_RELATIONSHIPS] = {};
        meta[this.PARAM_CHARACTERS] = characters;
        label_dict = {}
        for i, char in enumerate(characters):
            # print(i, char)
            label_dict[char] = i;
        meta[this.PARAM_LABEL_DICT] = label_dict;
        for ch in characters:
            chid = meta[this.PARAM_LABEL_DICT][ch];
            meta[this.PARAM_MASTER][chid] = chid;
            meta[this.PARAM_SERVANTS][chid] = set();
            meta[this.PARAM_FOES][chid] = set(); # still unused for now
        return meta;
    def finalize(this,meta):
        for ch in meta[this.PARAM_CHARACTERS]:
            chid = meta[this.PARAM_LABEL_DICT][ch];
            mid = meta[this.PARAM_MASTER][chid];
            meta[this.PARAM_RELATIONSHIPS][chid] = meta[this.PARAM_SERVANTS][mid].union({mid}).union(meta[this.PARAM_FOES][mid]);
        return meta;

    def add_masters(this,meta, servants, masters):
        for i in range(len(servants)):
            if (masters[i] not in meta[this.PARAM_CHARACTERS]):
                print("we have rough character", servants[i], "->-", masters[i])
                continue;
            try:
                sid = meta[this.PARAM_LABEL_DICT][servants[i]];
                mid = meta[this.PARAM_LABEL_DICT][masters[i]];
                meta[this.PARAM_MASTER][sid] = mid;
                meta[this.PARAM_SERVANTS][mid].add(sid);
            except:
                pass;
        return meta;

    def mount_protos(this,meta,):
        pass;


    #
    # meta=torch.load(mlt_full_meta_path);
    # meta=refactor_meta(meta);
    # meta=add_masters(meta,ks,vs);
    # meta=finalize(meta);
    #
    # torch.save(meta,mlt_full_fff_meta_path)


class drawer_mk3_cntr:
    def __init__(this,output_size,fntsize=32):
        this.fntsize=fntsize;
        this.output_size=output_size;
        this.spaces=[];


    def get_mask_core(this,what,font):
        try:
            font = ImageFont.truetype(font, this.fntsize);
        except:
            return None, None;
        mask = font.getmask(what);
        h,w=mask.size[1], mask.size[0];
        if(min(h,w)<=0):
            # cs=max(font.getbbox(what));
            # img = np.zeros([cs*2,cs*2, 3], dtype=np.uint8);
            # img = Image.fromarray(img);
            # draw = ImageDraw.Draw(img);
            # draw.text((cs//6,cs//6), what, (255, 255, 255), font=font)
            # assert (np.array(img).max()==0);
            return None,None;

        im = np.asarray(mask).reshape([h,w]).astype(np.uint8);
        if(im.max()==0):
            return None,None;
        return im,(h,w);

    def draw(this,what,font):
        valid,(h,w)=this.get_mask_core(what,font);

        if(valid is None):
            print("found space");
            this.spaces.append(what);
            print(what);
            return None,False;
        if(h > this.output_size*0.8 or w> this.output_size*0.8 ):
            if(h>w):
                scale=this.output_size*0.8/h;
            else:
                scale=this.output_size*0.8/w;
            ns=(int(w*scale),int(h*scale));
            try:
                valid=cv2.resize(valid,ns);
            except:
                pass;

            w=ns[0];
            h=ns[1];
        l=int((this.output_size-w)//2);
        t=int((this.output_size-h)//2);
        # print(l,t,"#",w,h);
        img= np.zeros([this.output_size,this.output_size,3],dtype=np.uint8)
        img[t:t+h,l:l+w,:]=valid.reshape(h,w,1);
        rim= cv2.resize(img,(this.output_size,this.output_size));
        flag=True;
        if(rim.max()<10):
            flag=False;
            print("wth???")
        return rim,flag;


    def draw_fallbacklist(this, what, fonts, css):
        for font, cs in zip(fonts, css):
            if (what in cs):
                img, flag = this.draw(what, font);
                if (flag):
                    return img,True;
        img = this.draw(what, fonts[0]);
        return img,False
class drawer_mk3_natural_loc(drawer_mk3_cntr):

    def draw(this,what,font):
        try:
            font = ImageFont.truetype(font, this.fntsize);
        except:
            this.spaces.append(what);
            return None,False;

        mask = font.getmask(what);
        h,w=mask.size[1], mask.size[0];
        if(min(h,w)<=0):
            this.spaces.append(what);
            return None, False;
        l,t,r,b=font.getbbox(what);
        ascent, descent = font.getmetrics()
        h=b;
        w=r;
        rcs=max(h,w);
        exm=int(rcs*0.2);
        cs=rcs+exm;
        offs=(exm+0,0);
        c=Image.new('RGB',(cs,cs) );
        image_draw=ImageDraw.Draw(c);
        image_draw.text(offs, what, (255, 255, 255), font);
        im = np.asarray(c).astype(np.uint8);
        if(im.max()==0):
            this.spaces.append(what);
            return None, False;

        valid =cv2.resize(im,(this.output_size,this.output_size));
        return valid,True;



        # As per a kind reviewer suggests we should use more fonts....
        # Welcome to claim your idea so we can add a thank you note in our repo.
        # Now this can be  a ram killer, but let's not store it to lmdb for now as we do not have fast disks either...

class render_lite_mk3_cntr:
    def set_drawer(this,outsize):
        this.drawer=drawer_mk3_cntr(outsize);
    def __init__(this,outsize):
        this.weird = [];
        this.set_drawer(outsize);

    def render_coreg3_mem(this, charset, fonts, font_ids, save_clip=False):
        magic = {}

        chars = [];
        protos = [];

        magic["chars"] = chars;
        magic["protos"] = protos;

        for i in tqdm.tqdm(range(len(charset))):
            ch = charset[i];
            protol_list = [];
            for fid in font_ids[i]:
                font = fonts[font_ids[i][fid]];
                protol, flag = this.drawer.draw(ch, font);
                if (flag):
                    protol_list.append(torch.tensor([protol[:, :, 0:1]]).float().permute(0, 3, 1, 2).contiguous());
                if (save_clip):
                    cv2.imwrite("im" + str(ord(ch[0])) + str(fid) + ".jpg", protol);
            chars.append(ch);
            protos.append(protol_list);
        return magic;

    def render_core_with_fallback(this, charset, fonts, font_ids, save_clip=False):
        magic = {}

        chars = [];
        protos = [];
        css = [fntmgmt.get_charset(F) for F in fonts];

        magic["chars"] = chars;
        magic["protos"] = protos;


        for i in range(len(charset)):
            font_list = [fonts[id] for id in font_ids[i]];
            css = [css[id] for id in font_ids[i]];
            ch = charset[i];
            protol,flag = this.drawer.draw(ch, font_list, css);
            if (save_clip):
                cv2.imwrite("im" + str(ord(ch[0])) + ".jpg", protol);
            chars.append(ch);
            if (i % 500 == 0):
                print(i, "of", len(charset));
            protos.append(torch.tensor([protol[:, :, 0:1]]).float().permute(0, 3, 1, 2).contiguous());
        return magic;






class render_lite_mk3_nl(render_lite_mk3_cntr):
    def set_drawer(this,outsize):
        this.drawer=drawer_mk3_natural_loc(outsize);




if __name__ == '__main__':
    # charset=u"QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm表1234567890";
    # sp_tokens=["[GO]","[s]"];
    from neko_sdk.ocr_modules.fontkit.fntmgmt import fntmgmt;
    import tqdm

    # F="/home/lasercat/HanaMinA.ttf";
    # cs=fntmgmt.get_charset(F);

    rlt = render_lite_mk3_nl(32);
    FS = ["/home/lasercat/allcjk/PlangothicP1-Regular.ttf", "/home/lasercat/allcjk/PlangothicP2-Regular.ttf",
          "/run/media/lasercat/f3a1698e-80ad-4473-8fc6-4df8c81c3831/rawdata/fonts/NotoSansCJK-Regular.ttc"];

    # FS += ["/home/lasercat/HanaMinA.ttf", "/home/lasercat/HanaMinB.ttf","gw2696945.ttf"];
    css = [fntmgmt.get_charset(F) for F in FS];
    im, flag = rlt.drawer.draw_fallbacklist("f", FS, css);
    cv2.imshow("meowg", im);
    cv2.waitKey(0);
    im, flag = rlt.drawer.draw_fallbacklist("_", FS, css);
    cv2.imshow("meowg", im);
    cv2.waitKey(0);
    #
    #
    im, flag = rlt.drawer.draw_fallbacklist("g", FS, css);
    cv2.imshow("meowg", im);
    cv2.waitKey(0);
    #
    im, flag = rlt.drawer.draw_fallbacklist(",", FS, css);
    cv2.imshow("meowg", im);
    cv2.waitKey(0);
    #
    im, flag = rlt.drawer.draw_fallbacklist("\'", FS, css);

    cv2.imshow("meowg", im);
    cv2.waitKey(0);
    cs="𪥶Α α, Β β, Γ γ, Δ δ, Ε ε, Ζ ζ, Η η, Θ θ, Ι ι, Κ κ, Λ λ, Μ μ, Ν ν, Ξ ξ, Ο ο, Π π, Ρ ρ, Σ σ/ς, Τ τ, Υ υ, Φ φ, Χ χ, Ψ ψ, Ω ω";
    for c in cs:
        im,flag = rlt.drawer.draw_fallbacklist(c, FS, css);
        if (flag):
            cv2.imshow("meowg", im);
            cv2.waitKey(300);
        else:
            # cv2.imshow("meowb", im);
            # cv2.waitKey(10);
            print(c);
    cv2.waitKey(0);

    # rlt.render(charset,sp_tokens,"support.pt");
    # cv2.imwrite("im"+i+".jpg",im);





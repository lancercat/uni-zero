import cv2;
import numpy as np;
import torch;
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


class render_lite:
    def __init__(this,cs=320,os=128,fos=32):
        this.CS=cs;
        this.os=os;
        this.fos=fos;
        this.spaces=[];
        this.px=np.zeros([this.CS,this.CS,3],dtype=np.uint8);
        this.weird=[];
    def range_draw(this,sz,what,font_):
        img = np.zeros([this.CS,this.CS,3],dtype=np.uint8);
        img=Image.fromarray(img);
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        font = ImageFont.truetype(font_, sz,layout_engine=ImageFont.LAYOUT_RAQM);

        # draw.text((x, y),"Sample Text",(r,g,b))
        draw.text((this.CS*0.2, this.CS*0.2),what,(255,255,255),font=font)
        fg=np.array(img.getdata()).reshape(this.CS,this.CS,3);
        this.px=np.maximum(this.px,fg);
        img = np.zeros([256, 256, 3], dtype=np.uint8);
        idx1, idx2, _ = np.where(fg > 0);
        if (fg.max() < 13):
            print("found space");
            this.weird.append({0, 0, what, font_});

            return None;
        min1 = idx1.min();
        max1 = idx1.max() + 1;
        min2 = idx2.min();
        max2 = idx2.max() + 1;
        h = max1 - min1;
        w = max2 - min2;
        if(h>84 or w > 84):
            print("found weird");
            this.weird.append({h,w,what,font_});



    def center_draw(this,sz,what,font):
        img = np.zeros([this.CS,this.CS,3],dtype=np.uint8);
        img=Image.fromarray(img);
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        font = ImageFont.truetype(font, sz);

        # draw.text((x, y),"Sample Text",(r,g,b))
        draw.text((this.CS*0.2, this.CS*0.2),what,(255,255,255),font=font)
        fg=np.array(img.getdata()).reshape(this.CS,this.CS,3);
        img = np.zeros([this.os,this.os,3],dtype=np.uint8);
        idx1,idx2,_=np.where(fg>0);
        if(fg.max()<13):
            print("found space");
            this.spaces.append(what);
            print(what);
            return cv2.resize(img, (this.fos, this.fos));
        min1=idx1.min();
        max1=idx1.max()+1;
        min2=idx2.min();
        max2=idx2.max()+1;
        h=max1-min1;
        w=max2-min2;
        valid=fg[min1:max1,min2:max2,:];
        if(h > this.os*0.9 or w> this.os*0.9 ):
            if(h>w):
                scale=this.os*0.9/h;
            else:
                scale=this.os*0.9/w;
            ns=(int(w*scale),int(h*scale));
            try:
                a=fg[min1:max1,min2:max2,:].copy().astype(np.uint8);
                valid=cv2.resize(a,ns);
            except:
                pass;

            w=ns[0];
            h=ns[1];
        l=int((this.os-w)//2);
        t=int((this.os-h)//2);
        # print(l,t,"#",w,h);
        img[t:t+h,l:l+w,:]=valid;
        return cv2.resize(img,(this.fos,this.fos));


    def render_range(this,charset,sp_tokens,fonts,font_ids,meta_file,save_clip=False):

        magic = {}
        chars = [];
        protos = [];

        magic["chars"] = chars;
        magic["sp_tokens"] = sp_tokens;
        magic["protos"] = protos;

        for i in sp_tokens:
            protos.append(None)

        for i in range(len(charset)):
            font=fonts[font_ids[i]];
            ch=charset[i];
            protol = this.range_draw(64, ch,font);
            if(i%500==0):
                print(i,"of",len(charset));
                cv2.imwrite("px.jpg",this.px);
                torch.save(this.weird, "weird.pt")

    def render_core(this,charset,sp_tokens,fonts,font_ids,save_clip=False):
        magic = {}

        chars = [];
        protos = [];

        magic["chars"] = chars;
        magic["sp_tokens"] = sp_tokens;
        magic["protos"] = protos;

        for i in sp_tokens:
            protos.append(None)

        for i in range(len(charset)):
            font = fonts[font_ids[i]];
            ch = charset[i];
            protol = this.center_draw(64, ch, font);
            if (save_clip):
                cv2.imwrite("im" + str(ord(ch[0])) + ".jpg", protol);
            chars.append(ch);
            if (i % 500 == 0):
                print(i, "of", len(charset));
            protos.append(torch.tensor([protol[:, :, 0:1]]).float().permute(0, 3, 1, 2).contiguous());
        return magic;
    # As per a kind reviewer suggests we should use more fonts....
    # Welcome to claim your idea so we can add a thank you note in our repo.
    # Now this can be  a ram killer, but let's not store it to lmdb for now as we do not have fast disks either...
    def render_coreg2_mem(this,charset,sp_tokens,fonts,font_ids,save_clip=False):
        magic = {}

        chars = [];
        protos = [];

        magic["chars"] = chars;
        magic["sp_tokens"] = sp_tokens;
        magic["protos"] = protos;

        for i in sp_tokens:
            protos.append(None)

        for i in range(len(charset)):
            ch = charset[i];
            protol_list=[];
            for fid in font_ids[i]:
                font = fonts[font_ids[i][fid]];
                protol = this.center_draw(64, ch, font);
                protol_list.append(torch.tensor([protol[:, :, 0:1]]).float().permute(0, 3, 1, 2).contiguous());
                if (save_clip):
                    cv2.imwrite("im" + str(ord(ch[0])) +str(fid)+ ".jpg", protol);
            chars.append(ch);
            if (i % 500 == 0):
                print(i, "of", len(charset));
            protos.append(protol_list);
        return magic;

    # for ablation purpose.
    def render_core_scabl(this, charset, sp_tokens, fonts, font_ids, save_clip=False):
        magic = {}

        chars = [];
        protos = [];

        magic["chars"] = chars;
        magic["sp_tokens"] = sp_tokens;
        magic["protos"] = protos;

        for i in sp_tokens:
            protos.append(None)

        for i in range(len(charset)):
            font = fonts[font_ids[i]];
            ch = charset[i];
            protol = this.center_draw(64, ch.lower(), font);
            if (save_clip):
                cv2.imwrite("im" + str(ord(ch[0])) + ".jpg", protol);
            chars.append(ch);
            if (i % 500 == 0):
                print(i, "of", len(charset));
            protos.append(torch.tensor([protol[:, :, 0:1]]).float().permute(0, 3, 1, 2).contiguous());
        return magic;

    def render(this,charset,sp_tokens,fonts,font_ids,meta_file,save_clip=False):
        magic=this.render_core(charset,sp_tokens,fonts,font_ids,save_clip);
        torch.save(magic, meta_file);
class render_lite_mk2(render_lite):
    def get_mask_core(this,sz,what,font):
        try:
            font = ImageFont.truetype(font, sz);
        except:
            return None, None, None;
        mask = font.getmask(what);
        h,w=mask.size[1], mask.size[0];
        if(min(h,w)<=0):
            cs=max(font.getbbox(what));
            img = np.zeros([cs*2,cs*2, 3], dtype=np.uint8);
            img = Image.fromarray(img);
            draw = ImageDraw.Draw(img);
            draw.text((cs//6,cs//6), what, (255, 255, 255), font=font)
            assert (np.array(img).max()==0);
            return None,None,None;

        im = np.asarray(mask).reshape([h,w]).astype(np.uint8);
        if(im.max()==0):
            return None,None,None;
        return im,h,w;

    def center_draw(this,sz,what,font):
        valid,h,w=this.get_mask_core(sz,what,font);

        if(valid is None):
            print("found space");
            this.spaces.append(what);
            print(what);
            return None;
        if(h > this.os*0.8 or w> this.os*0.8 ):
            if(h>w):
                scale=this.os*0.8/h;
            else:
                scale=this.os*0.8/w;
            ns=(int(w*scale),int(h*scale));
            try:
                valid=cv2.resize(valid,ns);
            except:
                pass;

            w=ns[0];
            h=ns[1];
        l=int((this.os-w)//2);
        t=int((this.os-h)//2);
        # print(l,t,"#",w,h);
        img= np.zeros([this.os,this.os,3],dtype=np.uint8)
        img[t:t+h,l:l+w,:]=valid.reshape(h,w,1);
        rim= cv2.resize(img,(this.fos,this.fos));
        if(rim.max()<10):
            print("wth???")
        return rim;


if __name__ == '__main__':
    # charset=u"QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm表1234567890";
    # sp_tokens=["[GO]","[s]"];
    rlt=render_lite();
    im=rlt.center_draw(64,"ন্ত্র","/home/lasercat/ssddata/tmp/Mina-Regular.ttf");
    cv2.imshow("meow",im);
    cv2.waitKey(0);
    # rlt.render(charset,sp_tokens,"support.pt");
    #cv2.imwrite("im"+i+".jpg",im);

    
    
    

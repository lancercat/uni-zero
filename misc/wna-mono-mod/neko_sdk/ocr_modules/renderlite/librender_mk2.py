import cv2;
import numpy as np;
import torch;
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


class drawer:
    def __init__(this, cs=320, os=128, fos=32, fsz=64):
        this.CS = cs;
        this.os = os;
        this.fos = fos;
        this.fsz = fsz;
        this.spaces = [];
        this.px = np.zeros([this.CS, this.CS, 3], dtype=np.uint8);
        this.zero=np.zeros([this.fos, this.fos, 3],dtype=np.uint8)
    def paint_on_canvas(this,what,font_):
        img = np.zeros([this.CS, this.CS, 3], dtype=np.uint8);
        img = Image.fromarray(img);
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        font = ImageFont.truetype(font_, this.fsz, layout_engine=ImageFont.LAYOUT_RAQM);

        # draw.text((x, y),"Sample Text",(r,g,b))
        draw.text((this.CS * 0.2, this.CS * 0.2), what, (255, 255, 255), font=font,spacing=0)
        fg = np.array(img.getdata(),dtype=np.uint8).reshape(this.CS, this.CS, 3);

        return fg;
    def get_valid_region(this,fg):
        return this.zero,False;
    def draw(this, what, font_):
        fg = this.paint_on_canvas(what, font_);
        this.px = np.maximum(this.px, fg);
        if (fg.max() < 13):
            print("found space");
            return this.zero, False;
        return this.get_valid_region(fg);

    def draw_fallbacklist(this, what, fonts, css):
        for font, cs in zip(fonts, css):
            if (what in cs):
                img, flag = this.draw(what, font);
                if (flag):
                    return img,True;
        img, _ = this.draw(what, fonts[0]);
        return img,False


        # As per a kind reviewer suggests we should use more fonts....
        # Welcome to claim your idea so we can add a thank you note in our repo.
        # Now this can be  a ram killer, but let's not store it to lmdb for now as we do not have fast disks either...




class range_drawer(drawer):
    def get_valid_region(this,fg):
        spr=0.5
        l=int(this.CS*0.2-this.fsz*0.25);
        t = int(this.CS * 0.2 - this.fsz * 0);
        b= int(this.CS*0.2+this.fsz*1.5);
        r= int(this.CS*0.2+this.fsz*1.25);
        idx1, idx2, _ = np.where(fg > 0);
        min1 = min(idx1.min(),t);
        max1 = max(idx1.max() + 1,b);
        min2 =  min(idx2.min(),l);
        max2 = max(idx2.max() + 1,r);
        h = max1-min1;
        w = max2-min2;
        if (h >= this.os or w >=this.os):
            if (h > w):
                scale = this.os / h;
            else:
                scale = this.os / w;
            ns = (min(int(w * scale),this.os), min(int(h * scale),this.os));
            a = fg[min1:max1, min2:max2, :].copy().astype(np.uint8);
            valid = np.zeros([this.os,this.os,3],dtype=np.uint8);
            valid[:ns[1],:ns[0]]=cv2.resize(a, ns);
        else:
            ms = max(h, w);
            valid = fg[min1:min1+ms, min2:min2+ms, :];
        return cv2.resize(valid, (this.fos, this.fos)), True;


class center_drawer(drawer):
    def get_valid_region(this,fg):
        img = np.zeros([this.os, this.os, 3], dtype=np.uint8);
        idx1, idx2, _ = np.where(fg > 0);
        if (fg.max() < 13):
            print("found space");
            return cv2.resize(img, (this.fos, this.fos)), False;
        min1 = idx1.min();
        max1 = idx1.max() + 1;
        min2 = idx2.min();
        max2 = idx2.max() + 1;
        h = max1 - min1;
        w = max2 - min2;
        valid = fg[min1:max1, min2:max2, :];
        if (h > w):
            scale = this.os * 0.8 / h;
        else:
            scale = this.os * 0.8 / w;
        ns = (int(w * scale), int(h * scale));
        try:
            a = fg[min1:max1, min2:max2, :].copy().astype(np.uint8);
            valid = cv2.resize(a, ns);
        except:
            pass;
        w = ns[0];
        h = ns[1];
        l = int((this.os - w) // 2);
        t = int((this.os - h) // 2);
        # print(l,t,"#",w,h);
        img[t:t + h, l:l + w, :] = valid;
        return cv2.resize(img, (this.fos, this.fos)), True;


class render_lite_mk2_range:
    def set_drawer(this,cs, os, fos, fsz):
        this.drawer=range_drawer(cs,os,fos,fsz);
    def __init__(this, cs=320, os=128, fos=32, fsz=64):
        this.weird = [];
        this.set_drawer(cs,os,fos,fsz);

    def render_coreg2_mem(this, charset, sp_tokens, fonts, font_ids, save_clip=False):
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
            protol_list = [];
            for fid in font_ids[i]:
                font = fonts[font_ids[i][fid]];
                protol, flag = this.drawer.draw(ch, font);
                if (flag):
                    protol_list.append(torch.tensor([protol[:, :, 0:1]]).float().permute(0, 3, 1, 2).contiguous());
                if (save_clip):
                    cv2.imwrite("im" + str(ord(ch[0])) + str(fid) + ".jpg", protol);
            chars.append(ch);
            if (i % 500 == 0):
                print(i, "of", len(charset));
            protos.append(protol_list);
        return magic;

    def render_core(this, charset, sp_tokens, fonts, font_ids, save_clip=False):
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
            protol,flag = this.drawer.draw(ch, font);
            if (save_clip):
                cv2.imwrite("im" + str(ord(ch[0])) + ".jpg", protol);
            chars.append(ch);
            if (i % 500 == 0):
                print(i, "of", len(charset));
            protos.append(torch.tensor([protol[:, :, 0:1]]).float().permute(0, 3, 1, 2).contiguous());
        return magic;

    def render_core_with_fallback(this, charset, sp_tokens, fonts, font_ids, save_clip=False):
        magic = {}

        chars = [];
        protos = [];
        css = [fntmgmt.get_charset(F) for F in fonts];

        magic["chars"] = chars;
        magic["sp_tokens"] = sp_tokens;
        magic["protos"] = protos;

        for i in sp_tokens:
            protos.append(None)

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


    def render(this, charset, sp_tokens, fonts, font_ids, meta_file, save_clip=False):
        magic = this.render_core(charset, sp_tokens, fonts, font_ids, save_clip);
        torch.save(magic, meta_file);


    def render_with_fallback(this, charset, sp_tokens, fonts, font_ids, meta_file, save_clip=False):
        magic = this.render_core_with_fallback(charset, sp_tokens, fonts, font_ids, save_clip);
        torch.save(magic, meta_file);
class render_lite_mk2_center(render_lite_mk2_range):
    def set_drawer(this,cs, os, fos, fsz):
        this.drawer=center_drawer(cs,os,fos,fsz);


if __name__ == '__main__':
    # charset=u"QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm表1234567890";
    # sp_tokens=["[GO]","[s]"];
    from neko_sdk.ocr_modules.fontkit.fntmgmt import fntmgmt;
    import tqdm

    # F="/home/lasercat/HanaMinA.ttf";
    # cs=fntmgmt.get_charset(F);

    rlt = render_lite_mk2_range();
    FS = ["/home/lasercat/allcjk/PlangothicP1-Regular.ttf", "/home/lasercat/allcjk/PlangothicP2-Regular.ttf",
          "/home/lasercat/ssddata/synth_data/fnts/NotoSansCJK-Regular.ttc"];

    # FS += ["/home/lasercat/HanaMinA.ttf", "/home/lasercat/HanaMinB.ttf","gw2696945.ttf"];
    css = [fntmgmt.get_charset(F) for F in FS];
    im, flag = rlt.drawer.draw_fallbacklist("f", FS, css);
    cv2.imshow("meowg", im);
    cv2.waitKey(0);
    im, flag = rlt.drawer.draw_fallbacklist("_", FS, css);
    cv2.imshow("meowg", im);
    cv2.waitKey(0);


    im, flag = rlt.drawer.draw_fallbacklist("g", FS, css);
    cv2.imshow("meowg", im);
    cv2.waitKey(0);

    im, flag = rlt.drawer.draw_fallbacklist(",", FS, css);
    cv2.imshow("meowg", im);
    cv2.waitKey(0);

    im, flag = rlt.drawer.draw_fallbacklist("\'", FS, css);

    cv2.imshow("meowg", im);
    cv2.waitKey(0);
    with open("/home/lasercat/alphabet_MTH1200.txt", "r") as fp:
        for l in tqdm.tqdm(fp):
            for c in l.strip().split(" "):
                im,flag = rlt.drawer.draw_fallbacklist( c, FS, css);
                if (flag):
                    cv2.imshow("meowg", im);
                    cv2.waitKey(10);
                else:
                    cv2.imshow("meowb", im);
                    cv2.waitKey(10);
                    print(c);

    # rlt.render(charset,sp_tokens,"support.pt");
    # cv2.imwrite("im"+i+".jpg",im);





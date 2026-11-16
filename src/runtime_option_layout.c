/* Full-height 1920x1080 layout builder and gothic ScrollBarV styler for `/menu/optionsetting.gfx`.
 *
 * Expands both the root `System` menu (`OptionSetting_Top_74`, 7 items -> 7 slots, zero scrolling)
 * and the custom PC option pages (`_Option_ItemList_71` / `ControllSetting`, used by `Controls`,
 * `Display`, and `Effects`, 6 -> 11 OptionRow slots + `Item_11_0` Defaults) to a unified
 * `[32 .. 1048]` vertical envelope on the 1920x1080 Scaleform stage so top/bottom ornate crowns
 * never jump between `System` and submenus, `Display` (11) & `Effects` (11) fit on a single screen
 * without scrolling, and `Controls` (35) gets 11 visible rows + a full-height gothic bronze/gold
 * scrollbar crowned with FromSoftware's 30-frame pulsing golden arrows (`Sprite 64`). */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const uint8_t *p; size_t bit; } SwfBits;

static unsigned sb_read(SwfBits *b,int n) {
    unsigned v=0;
    for (int i=0;i<n;++i,++b->bit) v=v<<1 | (b->p[b->bit>>3]>>(7-(b->bit&7)) & 1);
    return v;
}
static int opt_swf_tag(const uint8_t *buf,size_t at,size_t end,unsigned *code,size_t *body,size_t *len,size_t *next) {
    if (at+2>end) return 0;
    unsigned h=(unsigned)buf[at] | ((unsigned)buf[at+1]<<8);
    *code=h>>6; *len=h&0x3f; *body=at+2;
    if (*len==0x3f) {
        if (*body+4>end) return 0;
        *len=(size_t)buf[*body] | ((size_t)buf[*body+1]<<8) | ((size_t)buf[*body+2]<<16) | ((size_t)buf[*body+3]<<24);
        *body+=4;
    }
    if (*len>end-*body) return 0;
    *next=*body+*len;
    return 1;
}
static const char *opt_place_name(const uint8_t *t,unsigned code,size_t len) {
    size_t p=code==70 ? 2 : 1;
    unsigned flags=t[0], flags2=code==70 ? t[1] : 0;
    p+=2;
    if (code==70 && (flags2&0x08)) { while (p<len && t[p]) ++p; ++p; }
    if (flags&0x02) p+=2;
    if (!(flags&0x20) || p>=len) return NULL;
    SwfBits b={t,p*8};
    if (flags&0x04) {
        if (sb_read(&b,1)) { int n=(int)sb_read(&b,5); sb_read(&b,n); sb_read(&b,n); }
        if (sb_read(&b,1)) { int n=(int)sb_read(&b,5); sb_read(&b,n); sb_read(&b,n); }
        int n=(int)sb_read(&b,5); sb_read(&b,n); sb_read(&b,n);
        p=(b.bit+7)/8;
    }
    if (flags&0x08) {
        b.bit=p*8;
        int add=(int)sb_read(&b,1), mult=(int)sb_read(&b,1), n=(int)sb_read(&b,4);
        if (mult) for (int i=0;i<4;++i) sb_read(&b,n);
        if (add) for (int i=0;i<4;++i) sb_read(&b,n);
        p=(b.bit+7)/8;
    }
    if (flags&0x10) p+=2;
    if (p>=len || !memchr(t+p,0,len-p)) return NULL;
    return (const char *)t+p;
}

/* Pure translation SWF MATRIX (HasScale=0, HasRotate=0, 16-bit signed twips -> 39 bits = 5 bytes). */
static void pack_mat_t16(uint8_t out[5],int tx,int ty) {
    const uint64_t v=((uint64_t)16<<33) |
                     ((uint64_t)((unsigned)tx&0xffffu)<<17) |
                     ((uint64_t)((unsigned)ty&0xffffu)<<1);
    out[0]=(uint8_t)(v>>32); out[1]=(uint8_t)(v>>24); out[2]=(uint8_t)(v>>16);
    out[3]=(uint8_t)(v>>8);  out[4]=(uint8_t)v;
}

/* Scale + Translation SWF MATRIX (HasScale=1, 19-bit signed 16.16 fixed [-4.0..+3.9999];
 * HasRotate=0; 16-bit signed twips -> 1+5+19+19 + 1+5+16+16 = 82 bits -> 11 bytes). */
static void pack_mat_st11(uint8_t out[11],int sx_fp,int sy_fp,int tx,int ty) {
    memset(out,0,11);
    size_t bit=0;
    #define WB(val,nb) do { \
        unsigned _v=(unsigned)(val); \
        for (int _i=(nb)-1;_i>=0;--_i,++bit) \
            if ((_v>>(unsigned)_i)&1u) out[bit>>3]|=(uint8_t)(1u<<(7-(bit&7))); \
    } while (0)
    WB(1,1); WB(19,5); WB(sx_fp&0x7ffff,19); WB(sy_fp&0x7ffff,19);
    WB(0,1); WB(16,5); WB(tx&0xffff,16);     WB(ty&0xffff,16);
    #undef WB
}

typedef struct { uint8_t *data; size_t len, cap; } ByteBuf;
static int bb_reserve(ByteBuf *b,size_t add) {
    if (b->len+add<=b->cap) return 1;
    size_t ncap=b->cap ? b->cap*2 : 4096;
    while (ncap<b->len+add) ncap*=2;
    uint8_t *p=realloc(b->data,ncap);
    if (!p) return 0;
    b->data=p; b->cap=ncap;
    return 1;
}
static int bb_put(ByteBuf *b,const void *src,size_t n) {
    if (!bb_reserve(b,n)) return 0;
    if (n) memcpy(b->data+b->len,src,n);
    b->len+=n;
    return 1;
}
static int bb_put_tag(ByteBuf *b,unsigned code,const void *payload,size_t plen) {
    if (plen<63) {
        const unsigned h=(code<<6)|(unsigned)plen;
        uint8_t hdr[2]={(uint8_t)(h&0xff),(uint8_t)(h>>8)};
        return bb_put(b,hdr,2) && bb_put(b,payload,plen);
    }
    const unsigned h=(code<<6)|0x3fu;
    uint8_t hdr[6]={
        (uint8_t)(h&0xff),(uint8_t)(h>>8),
        (uint8_t)(plen&0xff),(uint8_t)((plen>>8)&0xff),
        (uint8_t)((plen>>16)&0xff),(uint8_t)((plen>>24)&0xff)
    };
    return bb_put(b,hdr,6) && bb_put(b,payload,plen);
}
static int emit_show_frame(ByteBuf *b) { return bb_put_tag(b,1,NULL,0); }
static int emit_end_tag(ByteBuf *b)    { return bb_put_tag(b,0,NULL,0); }
static int emit_frame_label(ByteBuf *b,const char *lbl) { return bb_put_tag(b,43,lbl,strlen(lbl)+1); }

/* Emits PlaceObject2 (code=26) or PlaceObject3 with BlendMode (code=70, blend>0). */
static int emit_place(ByteBuf *b,unsigned depth,int cid,const uint8_t *mat,size_t mat_len,
                      const uint8_t *cx,size_t cx_len,const char *name,uint8_t blend,int is_move) {
    uint8_t tmp[96];
    size_t p=0;
    uint8_t flags=(uint8_t)((is_move ? 0x01 : 0) | (cid>=0 ? 0x02 : 0) | (mat ? 0x04 : 0) |
                            (cx_len ? 0x08 : 0) | (name ? 0x20 : 0));
    if (blend) {
        tmp[p++]=flags;
        tmp[p++]=0x02; /* HasBlendMode */
    } else {
        tmp[p++]=flags;
    }
    tmp[p++]=(uint8_t)(depth&0xff);
    tmp[p++]=(uint8_t)(depth>>8);
    if (cid>=0) {
        tmp[p++]=(uint8_t)((unsigned)cid&0xff);
        tmp[p++]=(uint8_t)((unsigned)cid>>8);
    }
    if (mat && mat_len) { memcpy(tmp+p,mat,mat_len); p+=mat_len; }
    if (cx && cx_len)   { memcpy(tmp+p,cx,cx_len);   p+=cx_len; }
    if (name) { const size_t nl=strlen(name)+1; memcpy(tmp+p,name,nl); p+=nl; }
    if (blend) tmp[p++]=blend;
    return bb_put_tag(b,blend ? 70u : 26u,tmp,p);
}

/* Replaces DefineSprite `target_sid` in `src` with `new_body` (`frame_count` frames). */
static int replace_define_sprite(const uint8_t *src,size_t size,unsigned target_sid,unsigned frame_count,
                                 const uint8_t *new_tags,size_t new_tags_len,uint8_t **out_buf,size_t *out_size) {
    const size_t rect=(5+4*(src[8]>>3)+7)/8, first=8+rect+4;
    unsigned code; size_t body,len,next;
    for (size_t at=first; opt_swf_tag(src,at,size,&code,&body,&len,&next) && code; at=next) {
        if (code!=39 || len<4 || ((unsigned)src[body] | ((unsigned)src[body+1]<<8))!=target_sid) continue;
        ByteBuf out={0};
        if (!bb_put(&out,src,at)) { free(out.data); return 0; }
        const size_t sp_len=4+new_tags_len;
        uint8_t sp_hdr[4]={
            (uint8_t)(target_sid&0xff),(uint8_t)(target_sid>>8),
            (uint8_t)(frame_count&0xff),(uint8_t)(frame_count>>8)
        };
        const unsigned th=(39u<<6)|0x3fu;
        uint8_t tag_hdr[6]={
            (uint8_t)(th&0xff),(uint8_t)(th>>8),
            (uint8_t)(sp_len&0xff),(uint8_t)((sp_len>>8)&0xff),
            (uint8_t)((sp_len>>16)&0xff),(uint8_t)((sp_len>>24)&0xff)
        };
        if (!bb_put(&out,tag_hdr,6) || !bb_put(&out,sp_hdr,4) ||
            !bb_put(&out,new_tags,new_tags_len) || !bb_put(&out,src+next,size-next)) {
            free(out.data); return 0;
        }
        out.data[4]=(uint8_t)(out.len&0xff);
        out.data[5]=(uint8_t)((out.len>>8)&0xff);
        out.data[6]=(uint8_t)((out.len>>16)&0xff);
        out.data[7]=(uint8_t)((out.len>>24)&0xff);
        *out_buf=out.data;
        *out_size=out.len;
        return 1;
    }
    return 0;
}

/* FromSoftware's exact 19-frame FadeIn / Loop / FadeOut curve for OptionSetting pages. */
typedef struct {
    int active;
    int dy_twips;
    uint8_t item_cx[6]; size_t item_cx_len;
    uint8_t bg_cx[6];   size_t bg_cx_len;
} AnimKey;

static const AnimKey kOptionAnim[19]={
    {0,   0, {0},0, {0},0},
    {1, 200, {0x69,0x00,0x40,0x10,0x00,0x00},6, {0x69,0x00,0x40,0x10,0x00,0x00},6},
    {1, 160, {0x69,0x00,0x40,0x10,0x00,0xcc},6, {0x69,0x00,0x40,0x10,0x00,0x7c},6},
    {1, 120, {0x69,0x00,0x40,0x10,0x01,0x98},6, {0x69,0x00,0x40,0x10,0x00,0xf8},6},
    {1,  80, {0x69,0x00,0x40,0x10,0x02,0x68},6, {0x69,0x00,0x40,0x10,0x01,0x70},6},
    {1,  40, {0x69,0x00,0x40,0x10,0x03,0x34},6, {0x69,0x00,0x40,0x10,0x01,0xec},6},
    {1,   0, {0x04},1,                           {0x69,0x00,0x40,0x10,0x02,0x68},6},
    {0,0,{0},0,{0},0}, {0,0,{0},0,{0},0}, {0,0,{0},0,{0},0},
    {0,0,{0},0,{0},0}, {0,0,{0},0,{0},0}, {0,0,{0},0,{0},0},
    {1,  33, {0x69,0x00,0x40,0x10,0x03,0x54},6, {0x69,0x00,0x40,0x10,0x02,0x00},6},
    {1,  67, {0x69,0x00,0x40,0x10,0x02,0xac},6, {0x69,0x00,0x40,0x10,0x01,0x9c},6},
    {1, 100, {0x69,0x00,0x40,0x10,0x02,0x00},6, {0x69,0x00,0x40,0x10,0x01,0x34},6},
    {1, 133, {0x69,0x00,0x40,0x10,0x01,0x54},6, {0x69,0x00,0x40,0x10,0x00,0xcc},6},
    {1, 167, {0x69,0x00,0x40,0x10,0x00,0xac},6, {0x69,0x00,0x40,0x10,0x00,0x68},6},
    {1, 200, {0x69,0x00,0x40,0x10,0x00,0x00},6, {0x69,0x00,0x40,0x10,0x00,0x00},6},
};

typedef struct {
    unsigned depth;
    int cid;
    int scaled;
    int sx_fp, sy_fp;
    int tx, ty;
    const char *name;
    uint8_t blend;
} AnimChild;

static int build_19frame_menu_sprite(unsigned bg_cid,int bg_tx,int bg_ty,
                                     const AnimChild *kids,size_t nkids,ByteBuf *tags) {
    uint8_t bg_mat[11], m5[5], m11[11];
    static const uint8_t bg_f0_cx[6]={0x69,0x00,0x40,0x10,0x02,0x68};
    pack_mat_st11(bg_mat,65534,65533,bg_tx,bg_ty);
    if (!emit_place(tags,1,(int)bg_cid,bg_mat,11,bg_f0_cx,6,NULL,0,0)) return 0;
    for (size_t i=0;i<nkids;++i) {
        const AnimChild *k=&kids[i];
        if (k->scaled) {
            pack_mat_st11(m11,k->sx_fp,k->sy_fp,k->tx,k->ty);
            if (!emit_place(tags,k->depth,k->cid,m11,11,NULL,0,k->name,k->blend,0)) return 0;
        } else {
            pack_mat_t16(m5,k->tx,k->ty);
            if (!emit_place(tags,k->depth,k->cid,m5,5,NULL,0,k->name,k->blend,0)) return 0;
        }
    }
    if (!emit_show_frame(tags)) return 0;

    for (int f=1;f<=18;++f) {
        if (f==1 && !emit_frame_label(tags,"FadeIn")) return 0;
        if (f==6 && !emit_frame_label(tags,"Loop")) return 0;
        if (f==12 && !emit_frame_label(tags,"FadeOut")) return 0;
        if (kOptionAnim[f].active) {
            const AnimKey *a=&kOptionAnim[f];
            if (!emit_place(tags,1,-1,NULL,0,a->bg_cx,a->bg_cx_len,NULL,0,1)) return 0;
            for (size_t i=0;i<nkids;++i) {
                const AnimChild *k=&kids[i];
                if (k->scaled) {
                    pack_mat_st11(m11,k->sx_fp,k->sy_fp,k->tx,k->ty+a->dy_twips);
                    if (!emit_place(tags,k->depth,-1,m11,11,a->item_cx,a->item_cx_len,NULL,0,1)) return 0;
                } else {
                    pack_mat_t16(m5,k->tx,k->ty+a->dy_twips);
                    if (!emit_place(tags,k->depth,-1,m5,5,a->item_cx,a->item_cx_len,NULL,0,1)) return 0;
                }
            }
        }
        if (!emit_show_frame(tags)) return 0;
    }
    return emit_end_tag(tags);
}

/* Upgrades `ScrollBarV` (`sb_cid`) to Bloodborne Inventory's Antique Bronze (`#a1895f`) +
 * Warm Parchment Gold (`#e8dcb2`) + pulsing golden gothic arrows (`GothicUp` / `GothicDown`). */
static int style_gothic_scrollbar(const uint8_t *src,size_t size,unsigned sb_cid,unsigned arrow_cid,
                                  uint8_t **out_buf,size_t *out_size) {
    const size_t rect=(5+4*(src[8]>>3)+7)/8, first=8+rect+4;
    unsigned code; size_t body,len,next;
    long bar_sub=-1, track_sub=-1, bar_shape=-1, track_shape=-1;
    for (size_t at=first; opt_swf_tag(src,at,size,&code,&body,&len,&next) && code; at=next) {
        if (code!=39 || len<4 || ((unsigned)src[body] | ((unsigned)src[body+1]<<8))!=sb_cid) continue;
        size_t inner,ilen,inext; unsigned c;
        for (size_t t=body+4; opt_swf_tag(src,t,next,&c,&inner,&ilen,&inext) && c; t=inext) {
            if ((c==26 || c==70) && ilen>=5 && (src[inner]&0x02)) {
                const size_t co=(c==70 ? 4u : 3u);
                const unsigned cid=(unsigned)src[inner+co] | ((unsigned)src[inner+co+1]<<8);
                const char *nm=opt_place_name(src+inner,c,ilen);
                if (nm && !strcmp(nm,"GothicUp")) return 0;
                if (nm && !strcmp(nm,"Bar")) bar_sub=(long)cid;
                else if (!nm) track_sub=(long)cid;
            }
        }
    }
    for (size_t at=first; opt_swf_tag(src,at,size,&code,&body,&len,&next) && code; at=next) {
        if (code!=39 || len<4) continue;
        const unsigned sid=(unsigned)src[body] | ((unsigned)src[body+1]<<8);
        if ((long)sid!=bar_sub && (long)sid!=track_sub) continue;
        size_t inner,ilen,inext; unsigned c;
        for (size_t t=body+4; opt_swf_tag(src,t,next,&c,&inner,&ilen,&inext) && c; t=inext) {
            if ((c==26 || c==70) && ilen>=5 && (src[inner]&0x02)) {
                const size_t co=(c==70 ? 4u : 3u);
                const unsigned cid=(unsigned)src[inner+co] | ((unsigned)src[inner+co+1]<<8);
                if ((long)sid==bar_sub) bar_shape=(long)cid;
                else track_shape=(long)cid;
            }
        }
    }
    for (size_t at=first; opt_swf_tag(src,at,size,&code,&body,&len,&next) && code; at=next) {
        if (code!=39 || len<4 || ((unsigned)src[body] | ((unsigned)src[body+1]<<8))!=sb_cid) continue;
        const uint8_t up_mat[11]={0x88,0x32,0xc0,0x00,0x10,0x00,0x07,0x7e,0xc1,0xfe,0x70};
        const uint8_t dn_mat[11]={0x88,0x32,0x40,0x00,0x30,0x00,0x07,0x02,0x30,0xd3,0x40};
        ByteBuf tags={0};
        size_t inner,ilen,inext; unsigned c;
        int inserted=0;
        for (size_t t=body+4; opt_swf_tag(src,t,next,&c,&inner,&ilen,&inext); t=inext) {
            if (c==1 && !inserted) {
                if (!emit_place(&tags,9,(int)arrow_cid,up_mat,11,NULL,0,"GothicUp",0,0) ||
                    !emit_place(&tags,11,(int)arrow_cid,dn_mat,11,NULL,0,"GothicDown",0,0)) {
                    free(tags.data); return 0;
                }
                inserted=1;
            }
            if (!bb_put(&tags,src+t,inext-t)) { free(tags.data); return 0; }
            if (c==0) break;
        }
        uint8_t *dst=NULL; size_t dsz=0;
        const int ok=replace_define_sprite(src,size,sb_cid,1,tags.data,tags.len,&dst,&dsz);
        free(tags.data);
        if (!ok) return 0;
        for (size_t a2=first; opt_swf_tag(dst,a2,dsz,&code,&body,&len,&next) && code; a2=next) {
            if ((code==2 || code==22 || code==32 || code==83) && len>=16) {
                const unsigned sid=(unsigned)dst[body] | ((unsigned)dst[body+1]<<8);
                if ((long)sid==bar_shape) {
                    static const uint8_t pat[6]={0x01,0x00,0xcc,0xcc,0xcc,0x00};
                    static const uint8_t rep[6]={0x01,0x00,0xe8,0xdc,0xb2,0x00};
                    for (size_t k=body+2;k+6<=next;++k)
                        if (!memcmp(dst+k,pat,6)) { memcpy(dst+k,rep,6); break; }
                } else if ((long)sid==track_shape) {
                    static const uint8_t pat[8]={0x0a,0x00,0x00,0x00,0xcc,0xcc,0xcc,0xff};
                    static const uint8_t rep[8]={0x14,0x00,0x00,0x00,0xa1,0x89,0x5f,0xff};
                    for (size_t k=body+2;k+8<=next;++k)
                        if (!memcmp(dst+k,pat,8)) { memcpy(dst+k,rep,8); break; }
                }
            }
        }
        *out_buf=dst; *out_size=dsz;
        return 1;
    }
    return 0;
}

int runtime_patch_optionsetting_gfx(const uint8_t *src,size_t size,uint8_t **out_buf,size_t *out_size) {
    if (size<16 || memcmp(src,"GFX",3)) return 0;
    const size_t rect=(5+4*(src[8]>>3)+7)/8, first=8+rect+4;
    unsigned code; size_t body,len,next;
    long top_sp=-1, ctrl_sp=-1, lang_sp=-1, sb_cid=-1, arrow_cid=-1;
    for (size_t at=first; opt_swf_tag(src,at,size,&code,&body,&len,&next) && code; at=next) {
        if (code==76 && len>=2) {
            size_t p=body+2;
            for (unsigned i=0,n=(unsigned)src[body] | ((unsigned)src[body+1]<<8); i<n && p+2<next; ++i) {
                const unsigned id=(unsigned)src[p] | ((unsigned)src[p+1]<<8);
                const char *nm=(const char *)src+p+2;
                if (!memchr(nm,0,next-p-2)) return 0;
                if (!strcmp(nm,"OptionSetting_fla.OptionSetting_Top_74")) top_sp=(long)id;
                else if (!strcmp(nm,"OptionSetting_fla._Option_ItemList_71")) ctrl_sp=(long)id;
                else if (!strcmp(nm,"OptionSetting_fla._LanguageSetting_54")) lang_sp=(long)id;
                p+=2+strlen(nm)+1;
            }
        } else if (code==39 && len>=4) {
            size_t inner,ilen,inext; unsigned c;
            for (size_t t=body+4; opt_swf_tag(src,t,next,&c,&inner,&ilen,&inext) && c; t=inext) {
                if ((c==26 || c==70) && ilen>=5 && (src[inner]&0x02) && !(c==70 && (src[inner+1]&0x08))) {
                    const char *nm=opt_place_name(src+inner,c,ilen);
                    const size_t co=(c==70 ? 4u : 3u);
                    if (nm && co+2<=ilen) {
                        const unsigned cid=(unsigned)src[inner+co] | ((unsigned)src[inner+co+1]<<8);
                        if (sb_cid<0 && !strcmp(nm,"ScrollBarV")) sb_cid=(long)cid;
                        else if (arrow_cid<0 && !strcmp(nm,"RightArrow")) arrow_cid=(long)cid;
                    }
                }
            }
        }
    }
    if (top_sp<0 || ctrl_sp<0 || lang_sp<0 || sb_cid<0 || arrow_cid<0) return 0;

    /* Extract child character IDs from `ctrl_sp` (138), `top_sp` (150), `lang_sp` (115), and their frames. */
    int bg_dim=-1, ctrl_frame=-1, statusbar=-1, opt_row=-1, def_row=-1, ctrl_title=-1;
    int top_frame=-1, top_row=-1, top_title=-1;
    int lang_frame=-1, lang_notice=-1, lang_title=-1;
    for (size_t at=first; opt_swf_tag(src,at,size,&code,&body,&len,&next) && code; at=next) {
        if (code!=39 || len<4) continue;
        const unsigned sid=(unsigned)src[body] | ((unsigned)src[body+1]<<8);
        if ((long)sid!=ctrl_sp && (long)sid!=top_sp && (long)sid!=lang_sp) continue;
        size_t inner,ilen,inext; unsigned c;
        for (size_t t=body+4; opt_swf_tag(src,t,next,&c,&inner,&ilen,&inext) && c && c!=1; t=inext) {
            if ((c!=26 && c!=70) || ilen<5 || !(src[inner]&0x02)) continue;
            const unsigned d=(unsigned)src[inner+(c==70 ? 2 : 1)] | ((unsigned)src[inner+(c==70 ? 3 : 2)]<<8);
            const unsigned cid=(unsigned)src[inner+(c==70 ? 4 : 3)] | ((unsigned)src[inner+(c==70 ? 5 : 4)]<<8);
            const char *nm=opt_place_name(src+inner,c,ilen);
            if ((long)sid==ctrl_sp) {
                if (d==1) bg_dim=(int)cid;
                else if (d==3) ctrl_frame=(int)cid;
                else if (nm && !strcmp(nm,"StatusBar")) statusbar=(int)cid;
                else if (nm && !strcmp(nm,"Item_0_0")) opt_row=(int)cid;
                else if (nm && !strcmp(nm,"Item_6_0")) def_row=(int)cid;
                else if (!nm && d>100) ctrl_title=(int)cid;
            } else if ((long)sid==top_sp) {
                if (d==3) top_frame=(int)cid;
                else if (nm && !strcmp(nm,"Item_0_0")) top_row=(int)cid;
                else if (!nm && d>60) top_title=(int)cid;
            } else {
                if (d==3) lang_frame=(int)cid;
                else if (d==56) lang_notice=(int)cid;
                else if (!nm && d>80) lang_title=(int)cid;
            }
        }
    }
    if (bg_dim<0 || ctrl_frame<0 || statusbar<0 || opt_row<0 || def_row<0 || ctrl_title<0 ||
        top_frame<0 || top_row<0 || top_title<0 ||
        lang_frame<0 || lang_notice<0 || lang_title<0) return 0;

    int cf_smoke=-1, cf_grad=-1, cf_top_t=-1, cf_bot_t=-1, cf_pillar=-1, cf_crown=-1;
    int tf_smoke=-1, tf_crown=-1, tf_rule=-1;
    int lf_grad=-1;
    for (size_t at=first; opt_swf_tag(src,at,size,&code,&body,&len,&next) && code; at=next) {
        if (code!=39 || len<4) continue;
        const int sid=(int)((unsigned)src[body] | ((unsigned)src[body+1]<<8));
        if (sid!=ctrl_frame && sid!=top_frame && sid!=lang_frame) continue;
        size_t inner,ilen,inext; unsigned c;
        for (size_t t=body+4; opt_swf_tag(src,t,next,&c,&inner,&ilen,&inext) && c && c!=1; t=inext) {
            if ((c!=26 && c!=70) || ilen<5 || !(src[inner]&0x02)) continue;
            const unsigned d=(unsigned)src[inner+(c==70 ? 2 : 1)] | ((unsigned)src[inner+(c==70 ? 3 : 2)]<<8);
            const int cid=(int)((unsigned)src[inner+(c==70 ? 4 : 3)] | ((unsigned)src[inner+(c==70 ? 5 : 4)]<<8));
            if (sid==ctrl_frame) {
                if (d==1) cf_smoke=cid;
                else if (d==11) cf_grad=cid;
                else if (d==17) cf_top_t=cid;
                else if (d==25) cf_bot_t=cid;
                else if (d==41) cf_pillar=cid;
                else if (d==43) cf_crown=cid;
            } else if (sid==top_frame) {
                if (d==1) tf_smoke=cid;
                else if (d==11) tf_crown=cid;
                else if (d==23) tf_rule=cid;
            } else {
                if (d==11) lf_grad=cid;
            }
        }
    }
    if (cf_smoke<0 || cf_grad<0 || cf_top_t<0 || cf_bot_t<0 || cf_pillar<0 || cf_crown<0 ||
        tf_smoke<0 || tf_crown<0 || tf_rule<0 || lf_grad<0) return 0;

    uint8_t *cur=NULL,*nxt=NULL; size_t cur_sz=0,nxt_sz=0;
    if (!style_gothic_scrollbar(src,size,(unsigned)sb_cid,(unsigned)arrow_cid,&cur,&cur_sz))
        return 0;

    static const uint8_t cx_2cc[6]={0x69,0x00,0x40,0x10,0x02,0xcc};
    static const uint8_t cx_134[6]={0x69,0x00,0x40,0x10,0x01,0x34};
    static const uint8_t cx_0cc[6]={0x69,0x00,0x40,0x10,0x00,0xcc};

    /* 1. Rebuild `ctrl_frame` (135): full-height PC sheet stage_y = [32 .. 1048] (for Controls/Display/Effects). */
    {
        ByteBuf ft={0};
        uint8_t m11[11], m5[5];
        pack_mat_st11(m11,65536,90440,-531*20-2,-210*20); /* sy=1.38 */
        emit_place(&ft,1,cf_smoke,m11,11,cx_2cc,6,NULL,0,0);
        pack_mat_st11(m11,65536,117795,-269*20-4,-2714);  /* sy=1.7974, stage_y=120..766 */
        emit_place(&ft,11,cf_grad,m11,11,cx_2cc,6,NULL,8,0);
        pack_mat_st11(m11,-65536,-117795,288*20+16,10206);
        emit_place(&ft,14,cf_grad,m11,11,cx_2cc,6,NULL,8,0);
        pack_mat_t16(m5,0,-2754);                          /* stage_y=118.0 */
        emit_place(&ft,17,cf_top_t,m5,5,cx_134,6,NULL,8,0);
        pack_mat_t16(m5,0,10086);                          /* stage_y=760.0 */
        emit_place(&ft,25,cf_bot_t,m5,5,cx_0cc,6,NULL,8,0);
        pack_mat_st11(m11,65536,-65536,0,12766);           /* sy=-1.0, stage_y=894.0 */
        emit_place(&ft,33,cf_bot_t,m11,11,cx_134,6,NULL,8,0);
        pack_mat_st11(m11,65536,89393,122,-2474);          /* sy=1.364, stage_y=132..754 */
        emit_place(&ft,41,cf_pillar,m11,11,cx_0cc,6,NULL,8,0);
        pack_mat_t16(m5,-4466,-4474);                      /* top crown: stage_y=32..80 */
        emit_place(&ft,43,cf_crown,m5,5,NULL,0,NULL,8,0);
        pack_mat_st11(m11,65536,-65536,-4466,15846);       /* bot crown: stage_y=1000..1048 */
        emit_place(&ft,49,cf_crown,m11,11,NULL,0,NULL,8,0);
        emit_show_frame(&ft);
        emit_end_tag(&ft);
        const int ok=replace_define_sprite(cur,cur_sz,(unsigned)ctrl_frame,1,ft.data,ft.len,&nxt,&nxt_sz);
        free(ft.data); free(cur);
        if (!ok) return 0;
        cur=nxt; cur_sz=nxt_sz;
    }

    /* 2. Rebuild `top_frame` (145): standard compact envelope stage_y = [151.70 .. 893.65]
     *    matching Environment / Brightness / Network (frame145_y = stage_y + 105.55). */
    {
        ByteBuf ft={0};
        uint8_t m11[11], m5[5];
        static const uint8_t cx_334[6]={0x69,0x00,0x40,0x10,0x03,0x34};
        static const uint8_t cx_198[6]={0x69,0x00,0x40,0x10,0x01,0x98};
        pack_mat_st11(m11,65536,85197,-1260,6431);         /* sy=1.30, stage_y=216.0 */
        emit_place(&ft,1,tf_smoke,m11,11,cx_334,6,NULL,0,0);
        pack_mat_t16(m5,248,5145);                         /* top crown: stage_y=151.70 */
        emit_place(&ft,11,tf_crown,m5,5,NULL,0,NULL,8,0);
        pack_mat_st11(m11,65536,-65536,248,19984);         /* bot crown: stage_y=893.65 */
        emit_place(&ft,17,tf_crown,m11,11,NULL,0,NULL,8,0);
        pack_mat_t16(m5,-2449,6805);                       /* top rule: stage_y=234.70 */
        emit_place(&ft,23,tf_rule,m5,5,cx_198,6,NULL,8,0);
        pack_mat_t16(m5,-2449,17179);                      /* bot rule: stage_y=753.40 */
        emit_place(&ft,27,tf_rule,m5,5,cx_198,6,NULL,8,0);
        emit_show_frame(&ft);
        emit_end_tag(&ft);
        const int ok=replace_define_sprite(cur,cur_sz,(unsigned)top_frame,1,ft.data,ft.len,&nxt,&nxt_sz);
        free(ft.data); free(cur);
        if (!ok) return 0;
        cur=nxt; cur_sz=nxt_sz;
    }

    /* 3. Rebuild `lang_frame` (110) & `lang_sp` (115, `Language`): align its outlier [125.7..918.65]
     *    frame to the exact [151.70 .. 893.65] envelope of Environment / Brightness / Network! */
    {
        ByteBuf ft={0};
        uint8_t m11[11], m5[5];
        pack_mat_t16(m5,-10622,-1291);                     /* smoke: stage_y=191.15 */
        emit_place(&ft,1,cf_smoke,m5,5,cx_2cc,6,NULL,0,0);
        pack_mat_t16(m5,-5384,-23);                        /* grad: stage_y=254.55 */
        emit_place(&ft,11,lf_grad,m5,5,cx_2cc,6,NULL,8,0);
        pack_mat_st11(m11,-65536,-65536,5776,3156);
        emit_place(&ft,14,lf_grad,m11,11,cx_2cc,6,NULL,8,0);
        pack_mat_t16(m5,0,-380);                           /* top rule: stage_y=236.70 */
        emit_place(&ft,17,cf_top_t,m5,5,cx_134,6,NULL,8,0);
        pack_mat_t16(m5,0,2676);                           /* mid rule: stage_y=389.50 */
        emit_place(&ft,25,cf_bot_t,m5,5,cx_0cc,6,NULL,8,0);
        pack_mat_st11(m11,65536,-65536,0,10104);           /* bot rule: stage_y=760.90 */
        emit_place(&ft,33,cf_bot_t,m11,11,cx_134,6,NULL,8,0);
        pack_mat_st11(m11,65536,17236,122,300);            /* sy=0.263, stage_y=270.70 */
        emit_place(&ft,41,cf_pillar,m11,11,cx_0cc,6,NULL,8,0);
        pack_mat_t16(m5,-4466,-2080);                      /* top crown: stage_y=151.70 */
        emit_place(&ft,43,cf_crown,m5,5,NULL,0,NULL,8,0);
        pack_mat_st11(m11,65536,-65536,-4466,12759);       /* bot crown: stage_y=893.65 */
        emit_place(&ft,49,cf_crown,m11,11,NULL,0,NULL,8,0);
        emit_show_frame(&ft);
        emit_end_tag(&ft);
        const int ok=replace_define_sprite(cur,cur_sz,(unsigned)lang_frame,1,ft.data,ft.len,&nxt,&nxt_sz);
        free(ft.data); free(cur);
        if (!ok) return 0;
        cur=nxt; cur_sz=nxt_sz;
    }
    {
        const AnimChild kids[7]={
            {3,  lang_frame,  0,0,0,  8957, 1020, NULL,        0},
            {53, statusbar,   0,0,0,  9568, 6701, "StatusBar", 0},
            {56, lang_notice, 0,0,0,  1605, 9918, NULL,        0},
            {58, def_row,     0,0,0, 10820, 5539, "Item_2_0",  0},
            {65, opt_row,     0,0,0,  1840, 2542, "Item_1_0",  0},
            {74, opt_row,     0,0,0,  1840, 1300, "Item_0_0",  0},
            {83, lang_title,  0,0,0,  4626,  -20, NULL,        0},
        };
        ByteBuf sp={0};
        if (!build_19frame_menu_sprite((unsigned)bg_dim,-10045,-4094,kids,7,&sp)) {
            free(sp.data); free(cur); return 0;
        }
        const int ok=replace_define_sprite(cur,cur_sz,(unsigned)lang_sp,19,sp.data,sp.len,&nxt,&nxt_sz);
        free(sp.data); free(cur);
        if (!ok) return 0;
        cur=nxt; cur_sz=nxt_sz;
    }

    /* 4. Rebuild `ctrl_sp` (138): 11 OptionRows (`Item_0_0..Item_10_0`, descending depths 108..68 so
     *    dropdown popups cover lower rows!) + `Item_11_0` (`Defaults` at depth 64) + `PortScrollBarV`. */
    {
        static const char *const kRowNames[12]={
            "Item_0_0","Item_1_0","Item_2_0","Item_3_0","Item_4_0","Item_5_0",
            "Item_6_0","Item_7_0","Item_8_0","Item_9_0","Item_10_0","Item_11_0"
        };
        AnimChild kids[16];
        size_t nk=0;
        kids[nk++]=(AnimChild){3,   ctrl_frame, 0,0,0,     8937,  980,  NULL,         0};
        kids[nk++]=(AnimChild){53,  statusbar,  0,0,0,     9568,  9444, "StatusBar",  0};
        for (int k=0;k<11;++k) {
            const unsigned d=(unsigned)(108-k*4);
            const int py=-1534 + k*1120; /* stage_y = 130.0 + k * 56.0 */
            kids[nk++]=(AnimChild){d, opt_row, 0,0,0, 1840, py, kRowNames[k], 0};
        }
        kids[nk++]=(AnimChild){64,  def_row,    0,0,0,     10820, 12666, kRowNames[11], 0};
        kids[nk++]=(AnimChild){117, ctrl_title, 0,0,0,     4626,  -2454, NULL,          8};
        kids[nk++]=(AnimChild){120, (int)sb_cid,1, 62259,106168, 18200,-974, "PortScrollBarV", 0};

        ByteBuf sp={0};
        if (!build_19frame_menu_sprite((unsigned)bg_dim,-10045,-4134,kids,nk,&sp)) {
            free(sp.data); free(cur); return 0;
        }
        const int ok=replace_define_sprite(cur,cur_sz,(unsigned)ctrl_sp,19,sp.data,sp.len,&nxt,&nxt_sz);
        free(sp.data); free(cur);
        if (!ok) return 0;
        cur=nxt; cur_sz=nxt_sz;
    }

    /* 5. Rebuild `top_sp` (150, `System` root menu): 7 tight native rows (`68.5 px` step vs vanilla `67.65 px`)
     *    inside the compact [151.70 .. 893.65] frame, zero scrollbar! */
    {
        static const char *const kTopNames[7]={
            "Item_0_0","Item_1_0","Item_2_0","Item_3_0","Item_4_0","Item_5_0","Item_6_0"
        };
        AnimChild kids[10];
        size_t nk=0;
        kids[nk++]=(AnimChild){3,  top_frame, 0,0,0, 506,  -8200, NULL,        0};
        for (int k=0;k<7;++k) {
            const unsigned d=(unsigned)(31+k*4);
            const int ty=-1689 + k*1370; /* stage_y = 220.0 + k * 68.50 (Text_0 at +50.7) */
            kids[nk++]=(AnimChild){d, top_row, 0,0,0, 1481, ty, kTopNames[k], 0};
        }
        kids[nk++]=(AnimChild){66, statusbar, 0,0,0, 5716,  4706, "StatusBar", 0};
        kids[nk++]=(AnimChild){69, top_title, 0,0,0, 927,  -2015, NULL,        8};

        ByteBuf sp={0};
        if (!build_19frame_menu_sprite((unsigned)bg_dim,-13754,-6090,kids,nk,&sp)) {
            free(sp.data); free(cur); return 0;
        }
        const int ok=replace_define_sprite(cur,cur_sz,(unsigned)top_sp,19,sp.data,sp.len,&nxt,&nxt_sz);
        free(sp.data); free(cur);
        if (!ok) return 0;
        cur=nxt; cur_sz=nxt_sz;
    }

    *out_buf=cur;
    *out_size=cur_sz;
    return 1;
}

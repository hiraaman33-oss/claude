const pptxgen = require('pptxgenjs');
const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
const s = pres.addSlide();
s.background = { color: 'FFFFFF' };
const DARK='1F3A4D', ACC='0F7C80', TINT='EEF6F6', MUTE='5B6B75';
const T=(t,o)=>s.addText(t,Object.assign({fontFace:'Calibri',margin:0,isTextBox:true},o));

T('ML-Raman Classification of Chromonic Liquid Crystals on Different Substrates',{x:0.5,y:0.25,w:12.33,h:0.55,fontFace:'Cambria',fontSize:23,bold:true,color:DARK});
T('Sunset Yellow (SSY) on PDMS, Gr-PDMS, SiO₂/Si and Gr-SiO₂/Si · 80 Raman spectra (20/class) · λ = 473 nm · Python (scikit-learn)',{x:0.5,y:0.8,w:12.33,h:0.32,fontSize:12.5,italic:true,color:MUTE});

// pipeline chips
const steps=['Raw spectra 450–1800 cm⁻¹','Savitzky–Golay + scaling','VIP feature selection','PLS-DA & SVM-RBF','CV · external · permutation test'];
const cw=[2.3,2.2,2.05,1.95,2.93]; let x=0.5;
steps.forEach((t,i)=>{
  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x,y:1.3,w:cw[i],h:0.42,fill:{color:TINT},line:{color:TINT},rectRadius:0.08});
  T([{text:(i+1)+'  ',options:{bold:true,color:ACC}},{text:t,options:{color:DARK}}],{x:x+0.1,y:1.3,w:cw[i]-0.2,h:0.42,fontSize:10.5,valign:'middle',align:'center'});
  x+=cw[i];
  if(i<steps.length-1){ s.addShape(pres.shapes.ISOSCELES_TRIANGLE,{x:x+0.06,y:1.44,w:0.1,h:0.14,rotate:90,fill:{color:ACC},line:{color:ACC}}); x+=0.225; }
});

// figures
const fy=1.95, fh=2.8;
const figs=[
 ['/tmp/claude-0/z/ppt/media/image22.png',3.54,'Raman spectra: visually indistinguishable'],
 ['/tmp/claude-0/z/ppt/media/image23.png',5.24,'4-class confusion matrices: PLS-DA (A), SVM-RBF (B)'],
 ['/tmp/claude-0/z/ppt/media/image26.png',2.81,'PLS-DA 3D scores: PDMS vs SiO₂/Si split'],
];
x=0.5;
figs.forEach(f=>{
  s.addImage({path:f[0],x,y:fy,w:f[1],h:fh});
  T(f[2],{x,y:fy+fh+0.05,w:f[1],h:0.3,fontSize:10.5,italic:true,color:MUTE,align:'center'});
  x+=f[1]+0.37;
});

// stats
const stats=[
 ['47.5%','4-class CV accuracy (both models)'],
 ['75%','Binary, mixed external test (SVM)'],
 ['90%','Graphene-only external test (PLS-DA)'],
 ['p = 0.001','Permutation test (999 permutations)'],
];
const sw=2.9, sg=0.243;
stats.forEach((st,i)=>{
  const xx=0.5+i*(sw+sg);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:xx,y:5.3,w:sw,h:0.95,fill:{color:'FFFFFF'},line:{color:'CFE0E1',width:1},rectRadius:0.08});
  T(st[0],{x:xx+0.15,y:5.33,w:sw-0.3,h:0.5,fontFace:'Cambria',fontSize:24,bold:true,color:ACC,valign:'middle'});
  T(st[1],{x:xx+0.15,y:5.83,w:sw-0.3,h:0.36,fontSize:11,color:DARK,valign:'top'});
});

// conclusion
s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:0.5,y:6.45,w:12.33,h:0.8,fill:{color:TINT},line:{color:TINT},rectRadius:0.08});
T([{text:'Conclusion: ',options:{bold:true,color:ACC}},
   {text:'ML reveals substrate- and graphene-dependent SSY self-organisation invisible by eye. The main discriminating axis is the substrate (PDMS vs SiO₂/Si); SVM-RBF suits mixed sets, PLS-DA graphene-coated sets.',options:{color:DARK}}],
 {x:0.7,y:6.48,w:11.93,h:0.74,fontSize:12.5,valign:'middle'});
pres.writeFile({fileName:'/home/user/claude/slides/ML_Raman_Summary_Figures.pptx'}).then(()=>console.log('ok'));

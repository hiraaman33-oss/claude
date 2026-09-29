const pptxgen = require('pptxgenjs');
const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
const s = pres.addSlide();
s.background = { color: 'FFFFFF' };
const DARK='1F3A4D', ACC='0F7C80', TINT='EEF6F6', MUTE='5B6B75';
const T=(t,o)=>s.addText(t,Object.assign({fontFace:'Calibri',margin:0,isTextBox:true},o));

T('ML-Raman Classification of Chromonic Liquid Crystals on Different Substrates',{x:0.5,y:0.3,w:12.3,h:0.6,fontFace:'Cambria',fontSize:23,bold:true,color:DARK});
T('Sunset Yellow (SSY) on PDMS, Gr-PDMS, SiO₂/Si and Gr-SiO₂/Si · 80 Raman spectra (20/class) · λ = 473 nm · Python (scikit-learn)',{x:0.5,y:0.92,w:12.3,h:0.35,fontSize:13,italic:true,color:MUTE});

// Pipeline row
T('What I did',{x:0.5,y:1.5,w:6,h:0.4,fontFace:'Cambria',fontSize:18,bold:true,color:DARK});
const steps=[
 ['Raw spectra','80 spectra, 450–1800 cm⁻¹'],
 ['Preprocessing','Savitzky–Golay smoothing, scaling on training folds only'],
 ['VIP selection','Top 1000–2200 variables'],
 ['Models','PLS-DA and SVM-RBF'],
 ['Validation','5-fold CV, external test, permutation test'],
];
const bw=2.26, gap=0.25;
steps.forEach((st,i)=>{
  const x=0.5+i*(bw+gap);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x,y:2.0,w:bw,h:1.45,fill:{color:TINT},line:{color:TINT},rectRadius:0.08});
  s.addShape(pres.shapes.OVAL,{x:x+0.15,y:2.13,w:0.4,h:0.4,fill:{color:ACC},line:{color:ACC}});
  T(String(i+1),{x:x+0.15,y:2.13,w:0.4,h:0.4,align:'center',valign:'middle',fontSize:13,bold:true,color:'FFFFFF'});
  T(st[0],{x:x+0.65,y:2.13,w:bw-0.75,h:0.4,valign:'middle',fontSize:14,bold:true,color:DARK});
  T(st[1],{x:x+0.15,y:2.63,w:bw-0.3,h:0.75,valign:'top',fontSize:11.5,color:MUTE});
  if(i<steps.length-1) s.addShape(pres.shapes.ISOSCELES_TRIANGLE,{x:x+bw+0.06,y:2.62,w:0.13,h:0.2,rotate:90,fill:{color:ACC},line:{color:ACC}});
});

// Key results stats
T('Key results',{x:0.5,y:3.75,w:6,h:0.4,fontFace:'Cambria',fontSize:18,bold:true,color:DARK});
const stats=[
 ['47.5%','4-class accuracy (both models): spectra overlap strongly'],
 ['75%','Binary PDMS vs SiO₂/Si, mixed external test (SVM-RBF)'],
 ['90%','Binary, graphene-only external test (PLS-DA)'],
 ['p = 0.001','Permutation test (999): signal is real, not chance'],
];
const sw=2.9, sg=0.23;
stats.forEach((st,i)=>{
  const x=0.5+i*(sw+sg);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x,y:4.25,w:sw,h:1.55,fill:{color:'FFFFFF'},line:{color:'CFE0E1',width:1},rectRadius:0.08});
  T(st[0],{x:x+0.2,y:4.35,w:sw-0.4,h:0.7,fontFace:'Cambria',fontSize:34,bold:true,color:ACC,valign:'middle'});
  T(st[1],{x:x+0.2,y:5.08,w:sw-0.4,h:0.65,fontSize:12,color:DARK,valign:'top'});
});

// Conclusion
s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:0.5,y:6.05,w:12.33,h:1.05,fill:{color:TINT},line:{color:TINT},rectRadius:0.08});
T([{text:'Conclusion: ',options:{bold:true,color:ACC}},
   {text:'ML reveals substrate- and graphene-dependent self-organisation of SSY that is invisible by eye. The main discriminating axis is the substrate (PDMS vs SiO₂/Si), and the best classifier depends on the samples: SVM-RBF for mixed sets, PLS-DA for graphene-coated sets.',options:{color:DARK}}],
 {x:0.75,y:6.12,w:11.85,h:0.9,fontSize:13.5,valign:'middle'});
pres.writeFile({fileName:'/home/user/claude/slides/ML_Raman_Summary.pptx'}).then(()=>console.log('ok'));

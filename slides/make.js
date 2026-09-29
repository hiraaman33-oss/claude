const pptxgen = require('pptxgenjs');
const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
const s = pres.addSlide();
s.background = { color: 'FFFFFF' };
const DARK='1F3A4D', ACC='0F7C80', TINT='EEF6F6', MUTE='5B6B75', LINE='D5E3E4';

s.addText('Training Activities: Mini-courses, Seminars & Conferences', {x:0.5,y:0.3,w:12.3,h:0.7,fontFace:'Cambria',fontSize:30,bold:true,color:DARK,margin:0,isTextBox:true});
s.addText('PhD Programme in Information Engineering – 39th Cycle · DIIES', {x:0.5,y:0.98,w:12.3,h:0.35,fontFace:'Calibri',fontSize:13,italic:true,color:MUTE,margin:0,isTextBox:true});

function head(x,y,w,num,title){
  s.addShape(pres.shapes.OVAL,{x,y,w:0.45,h:0.45,fill:{color:ACC},line:{color:ACC}});
  s.addText(num,{x,y,w:0.45,h:0.45,align:'center',valign:'middle',fontFace:'Calibri',fontSize:14,bold:true,color:'FFFFFF',margin:0,isTextBox:true});
  s.addText(title,{x:x+0.6,y,w:w-0.6,h:0.45,valign:'middle',fontFace:'Cambria',fontSize:17,bold:true,color:DARK,margin:0,isTextBox:true});
}
function list(x,y,w,h,items,fs){
  const runs=[];
  items.forEach((it,i)=>{
    runs.push({text:it[0],options:{bold:true,color:DARK,breakLine:true}});
    runs.push({text:it[1],options:{color:MUTE,fontSize:fs-1.5,breakLine:i<items.length-1,paraSpaceAfter:6}});
  });
  s.addText(runs,{x,y,w,h,valign:'top',fontFace:'Calibri',fontSize:fs,margin:0,isTextBox:true});
}

// Left column: mini-courses
const LX=0.5, LW=4.4;
s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:LX,y:1.6,w:LW,h:3.75,fill:{color:TINT},line:{color:TINT},rectRadius:0.1});
head(LX+0.25,1.78,LW-0.5,'1','Mini-courses (8 h each)');
list(LX+0.25,2.38,LW-0.5,2.9,[
 ['Edge Machine Learning for Data Analysis on Low Computational Capacity Devices','Prof. Merenda'],
 ['Ensuring Trustworthiness in Federated Learning','Prof. Fisichella'],
 ['Digital Identity','Prof. Lax'],
 ['Health Management','Prof. Romano'],
 ['Foundations of Quantum Mechanics and Applications','Profs. Messina and Faggio'],
],11.5);

// Left column: conference
s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:LX,y:5.55,w:LW,h:1.65,fill:{color:TINT},line:{color:TINT},rectRadius:0.1});
head(LX+0.25,5.72,LW-0.5,'3','Conference');
list(LX+0.25,6.27,LW-0.5,0.9,[
 ['NanoInnovation 2026 · Rome · 14–18 Sept 2026','Oral presentation: "Machine Learning Raman Classification of Chromonic Liquid Crystal Self-Organisation on Different Substrates"'],
],11.5);

// Right: seminars table
const RX=5.2, RW=7.63;
head(RX,1.6,RW,'2','Seminars');
const hdr=(t)=>({text:t,options:{bold:true,color:'FFFFFF',fill:{color:ACC}}});
const rows=[
 ['Some Results for Electromagnetic Design and Retrieval Problems','Prof. Mats Gustafsson','1.5 h'],
 ['Connected and Automated Mobility (MOST Spoke 6 dissemination activities)','M. Segata, I. Turcanu','3 h'],
 ['Factors Affecting the Port Times of Container Ships','—','4 h'],
 ['Resonances in Geometrically Finite Graphs','Dr Carsten Peterson','2 h'],
 ['Networking Solutions for Cooperative, Connected and Automated Mobility','Profs. C. Campolo, A. Bazzi, S. Sargento; Drs F. Brasca, G. Landi','4 h'],
 ['Digital Twins and Deep Learning: Advancing Infrastructure Sustainability and Resilience through Remote Sensing and Computer Vision','Dr Surya Sarat Chandra Congress','3 h'],
 ['Workshop','DICEAM','—'],
];
const data=[[hdr('Seminar'),hdr('Instructor(s)'),hdr('Hours')]];
rows.forEach((r,i)=>{
  const f={color: i%2? 'FFFFFF':TINT};
  data.push([{text:r[0],options:{fill:f,bold:true,color:DARK}},{text:r[1],options:{fill:f,color:MUTE}},{text:r[2],options:{fill:f,color:DARK,align:'center'}}]);
});
s.addTable(data,{x:RX,y:2.2,w:RW,colW:[4.25,2.63,0.75],fontFace:'Calibri',fontSize:12,valign:'middle',border:{type:'solid',pt:0.5,color:LINE},margin:[3,5,3,5]});
pres.writeFile({fileName:'Training_Activities.pptx'}).then(()=>console.log('ok'));

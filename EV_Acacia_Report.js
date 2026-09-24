// Builds EV_Acacia_Report.docx: literature background (plant EVs vs plant nanovesicles), experimental
// conditions and the Raman / SERS results for EV Acacia and EV nano Acacia.
// Run: NODE_PATH=<dir with node_modules> node EV_Acacia_Report.js
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, LevelFormat, PageBreak, Footer, PageNumber,
} = require("docx");

const W = 9026; // A4 content width (DXA) with 1-inch margins
const FONT = "Arial";

// ------------------------------------------------------------------ helpers
function runs(text, base = {}) {
  // **bold**, ^superscript^ and _{subscript} markup
  const out = [];
  const re = /(\*\*[^*]+\*\*|\^[^^]+\^|_\{[^}]+\})/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(...runs(t.slice(2, -2), { ...base, bold: true }));
    else if (t.startsWith("^")) out.push(new TextRun({ text: t.slice(1, -1), superScript: true, ...base }));
    else out.push(new TextRun({ text: t.slice(2, -1), subScript: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (t, o = {}) => new Paragraph({ children: runs(t, o.run || {}), spacing: { after: 120, line: 276 },
  alignment: o.align || AlignmentType.JUSTIFIED });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const B = (t, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: runs(t),
  spacing: { after: 60, line: 264 } });
const N = (t) => new Paragraph({ numbering: { reference: "refs", level: 0 }, children: runs(t, { size: 18 }),
  spacing: { after: 40 } });
const Caption = (t) => new Paragraph({ children: runs(t, { size: 18, italics: true, color: "404040" }),
  spacing: { before: 60, after: 200 }, alignment: AlignmentType.JUSTIFIED });

function Fig(path, wpx, hpx, maxW = 600) {
  const w = maxW, h = Math.round(maxW * hpx / wpx);
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width: w, height: h } })] });
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "A6A6A6" };
const borders = { top: border, bottom: border, left: border, right: border };
function Tbl(header, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const fs_ = opts.size || 17;
  const cell = (txt, w, head, fill) => new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: head ? { fill: "1F4E79", type: ShadingType.CLEAR, color: "auto" }
      : fill ? { fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: [new Paragraph({ children: runs(String(txt), { size: fs_, bold: head, color: head ? "FFFFFF" : undefined }) })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, widths[i], true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, widths[i], false,
        opts.fill ? opts.fill(r) : undefined)) })),
    ],
  });
}
const gap = () => new Paragraph({ children: [], spacing: { after: 80 } });
const statusFill = (r) => {
  const s = r.find((c) => typeof c === "string" && (c.startsWith("in both") || c.startsWith("only")));
  if (!s) return undefined;
  return s.startsWith("in both") ? "E2F0D9" : undefined;
};

// ------------------------------------------------------------------ content
const body = [];

// title page
body.push(
  new Paragraph({ spacing: { before: 2400, after: 240 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Raman and SERS characterisation of", size: 36, bold: true, color: "1F4E79" })] }),
  new Paragraph({ spacing: { after: 480 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "EV Acacia and EV nano Acacia", size: 44, bold: true, color: "1F4E79" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: runs(
    "Plant extracellular vesicles vs plant-derived nanovesicles: background, experimental conditions and data interpretation",
    { size: 24, italics: true }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 960 }, children: runs(
    "Spontaneous Raman on CaF_{2} and SERS on silver-decorated silicon nanowires (AgSiNW), 532 nm", { size: 22 }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, children: runs(
    "All spectral results are from raw data (no smoothing, no baseline correction)", { size: 20, color: "595959" }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, children: runs(
    "September 2026", { size: 20, color: "595959" }) }),
  new Paragraph({ children: [new PageBreak()] }),
);

// contents
body.push(
  new Paragraph({ children: [new TextRun({ text: "Contents", bold: true, size: 28, color: "1F4E79" })], spacing: { after: 120 } }),
  ...[
    "Summary",
    "1. Background",
    "    1.1 Plant extracellular vesicles vs plant-derived nanovesicles",
    "    1.2 Where they are used",
    "    1.3 Acacia as a source",
    "    1.4 Why this work was done",
    "2. Experimental conditions",
    "    2.1 Data processing",
    "3. Results",
    "    3.1 EV Acacia: CaF₂ vs AgSiNW",
    "    3.2 EV nano Acacia: CaF₂ vs AgSiNW",
    "    3.3 EV Acacia vs EV nano Acacia on CaF₂",
    "    3.4 EV Acacia vs EV nano Acacia on AgSiNW",
    "4. Interpretation",
    "    4.1 The two preparations are chemically different",
    "    4.2 Why the CaF₂ bands (1456, 1726 cm⁻¹) are missing on AgSiNW",
    "    4.3 What this means for possible uses",
    "5. Limitations",
    "6. Recommendations",
    "References"
  ].map((t) => new Paragraph({ spacing: { after: 60 }, indent: { left: t.startsWith(" ") ? 400 : 0 },
    children: [new TextRun({ text: t.trim(), bold: !t.startsWith(" "), size: 21 })] })),
  new Paragraph({ children: [new PageBreak()] }),
);

// summary
body.push(H1("Summary"));
body.push(P("Two vesicle preparations called **EV Acacia** and **EV nano Acacia** were received from another laboratory. " +
  "The isolation protocol, plant species and plant organ were not provided. They were measured by spontaneous " +
  "Raman spectroscopy on CaF_{2} and by surface-enhanced Raman spectroscopy (SERS) on AgSiNW substrates. The main findings are:"));
body.push(
  B("**EV Acacia on CaF_{2}** gives a reproducible fingerprint across five replicate spectra (r = 0.96–0.99). It is dominated by " +
    "CH_{2}/CH_{3} deformation at ~1456 cm^-1^, an ester C=O stretch at ~1729 cm^-1^ and CH stretching at ~2937 cm^-1^. " +
    "Weaker protein (amide III, Phe/Tyr) and polysaccharide bands are also present. This is a lipid- and ester-rich vesicle signature."),
  B("**EV nano Acacia on CaF_{2}** has a different fingerprint. Its strongest band is at 1096 cm^-1^ (C–O/C–C of carbohydrate, PO_{2}^-^, " +
    "gauche C–C), with a saccharide C–O–C band at 900 cm^-1^ and a CH band at ~2891 cm^-1^. The ester C=O band is **absent**. " +
    "Its correlation with the EV Acacia mean is only r = 0.77, against 0.96–0.99 between EV Acacia replicates."),
  B("**On AgSiNW (SERS)** neither sample reproduces its CaF_{2} fingerprint. The 1456 and 1726 cm^-1^ bands were never found together " +
    "in any of the 100 EV Acacia map spectra. SERS spectra of both samples are dominated by the Si phonon (519 cm^-1^), " +
    "Phe (1002 cm^-1^), carboxylate and broad amorphous-carbon D/G bands (~1380 and ~1580–1607 cm^-1^)."),
  B("This is explained by SERS hot-spot and orientation selectivity, a roughly 600-fold smaller photon budget in the AgSiNW " +
    "settings, and photo-induced carbon on silver. It is **not** evidence that lipids are absent. CaF_{2} is therefore the reliable " +
    "basis for comparing the two samples."),
  B("The CaF_{2} difference fits the literature distinction (Section 1). True EVs are secreted, membrane-rich vesicles. " +
    "Homogenisation-derived nanovesicles carry co-isolated biomass material; for Acacia this is abundant polysaccharide gums and polyphenols. " +
    "This remains a hypothesis until the provider lab confirms the isolation methods."),
);

// ------------------------------------------------------------------ 1 background
body.push(new Paragraph({ children: [new PageBreak()] }), H1("1. Background"));
body.push(H2("1.1 Plant extracellular vesicles vs plant-derived nanovesicles"));
body.push(P("Plants release membrane vesicles, and vesicle-like particles can also be made from plant tissue. Both are " +
  "often called \"plant exosomes\", but the literature now separates two different materials [1–3]:"));
body.push(
  B("**Plant-derived extracellular vesicles (PDEVs, \"true\" plant EVs).** Living cells actively secrete these into the " +
    "extracellular space (apoplast). They are usually isolated from apoplastic washing fluid (vacuum infiltration followed by " +
    "low-speed centrifugation), or from the medium of cell-suspension or callus cultures, then purified by ultracentrifugation, " +
    "density gradients or size-exclusion chromatography [2,4,5]. Their cargo is loaded selectively and includes stress- and " +
    "defence-related proteins, small RNAs and membrane lipids [1,4]. They take part in plant immunity and cross-kingdom RNA " +
    "communication with pathogens [1]."),
  B("**Plant-derived nanovesicles / nanoparticles (PDNVs, PDNPs, \"exosome-like nanoparticles\", ELNs).** These form when " +
    "tissue is juiced, blended or homogenised, and are then collected by differential centrifugation. Membranes of " +
    "many compartments (plasma membrane, vacuole, organelles) break and reseal. The particles therefore reflect the bulk " +
    "lipid and metabolite composition of the biomass rather than a selectively loaded cargo [1,3]. Yield is high, " +
    "which makes them attractive for applications. However, they can be contaminated by co-isolated soluble material and intracellular membranes [2,3]."),
);
body.push(P("Pinedo et al. pointed out that the two terms are often used interchangeably. They called for rigorous reporting of " +
  "the source and isolation method, because these determine what the particles are [2]. Table 1 summarises the differences."));
body.push(Tbl(["Property", "Plant EVs (PDEVs)", "Plant nanovesicles (PDNVs / ELNs)"], [
  ["Origin", "Actively secreted by living cells into the apoplast", "Formed from membranes broken during tissue disruption"],
  ["Typical source", "Apoplastic washing fluid; cell-culture or callus medium", "Juice or homogenate of fruit, root, rhizome, leaf"],
  ["Cargo", "Selective: defence proteins, sRNA, membrane lipids", "Bulk: reflects the lipid and metabolite composition of the tissue"],
  ["Yield / purity", "Low yield, higher purity", "High yield, more co-isolated proteins, metabolites and polysaccharides"],
  ["Main research use", "Plant immunity, cross-kingdom RNA transfer, engineered carriers", "Oral delivery, gut / anti-inflammatory therapy, drug carriers, cosmetics"],
  ["Key references", "[1,2,4,5]", "[1,3,6–9]"],
], [1800, 3613, 3613]));
body.push(Caption("Table 1. Plant EVs and plant-derived nanovesicles compared, based on the cited reviews and primary studies."));

body.push(H2("1.2 Where they are used"));
body.push(P("**Plant EVs** are mainly used in fundamental plant biology. Rutter and Innes isolated EVs from the leaf " +
  "apoplast of Arabidopsis and showed that they are enriched in stress-response proteins [4]. EVs are also being developed as " +
  "biological carriers. Kocholatá et al. compared tobacco EVs from apoplastic fluid, calli and suspension cultures and loaded " +
  "callus-derived vesicles with siRNA for delivery [5]."));
body.push(P("**Plant nanovesicles** are the most widely applied in biomedicine:"));
body.push(
  B("**Oral and gut applications.** Edible exosome-like nanoparticles from grapefruit, ginger, grape and carrot are taken up by " +
    "intestinal macrophages and stem cells and induce anti-inflammatory and protective genes. Their lipid composition depends on " +
    "the source, e.g. phosphatidic-acid-rich ginger vs phosphatidylcholine-rich grapefruit particles [6]."),
  B("**Inflammatory bowel disease.** Orally given ginger-derived nanoparticles reduced colitis and colitis-associated cancer in mice [7]."),
  B("**Anti-inflammatory activity.** Vesicle-like nanoparticles from honey inhibit the NLRP3 inflammasome and protect against " +
    "liver injury in mice [8]."),
  B("**Drug delivery, skin and cosmetics, microbiome.** Reviews describe PDNVs as natural carriers for drugs and nucleic acids, " +
    "as skin and cosmetic actives, and as modulators of the gut microbiota [1,9]."),
);

body.push(H2("1.3 Acacia as a source"));
body.push(P("Acacia nilotica (now Vachellia nilotica) is widely used in traditional medicine. Its main phytochemical classes are tannins, " +
  "flavonoids and other polyphenols, alkaloids, fatty acids and polysaccharide gums. Its extracts show anti-inflammatory, " +
  "antioxidant, antimicrobial, antidiarrhoeal and hypoglycaemic activity [10,11]. The exact Acacia species and organ used for " +
  "these samples were not provided. The Acacia literature is cited here as general context. Our literature search found no " +
  "published study of EVs or nanovesicles from Acacia. Vesicles from this plant are therefore a new subject, and a " +
  "label-free chemical fingerprint is a useful first characterisation."));

body.push(H2("1.4 Why this work was done"));
body.push(P("Raman spectroscopy gives a label-free chemical fingerprint of EVs: lipids, proteins, nucleic acids and carbohydrates " +
  "in a single measurement. It has been used to characterise and discriminate EV preparations [12,13]. CaF_{2} is a standard " +
  "low-background substrate for biological Raman spectroscopy [14], and SERS can raise sensitivity by orders of " +
  "magnitude for very small sample amounts [13,15]. The aims of this analysis were therefore:"));
body.push(
  B("to obtain a reliable Raman fingerprint of EV Acacia and EV nano Acacia on CaF_{2};"),
  B("to test whether SERS on AgSiNW with a very small amount of sample (1 µL drop; 20 ng for EV nano Acacia) reproduces this fingerprint, " +
    "in particular the lipid bands at 1456 and 1726 cm^-1^;"),
  B("to find out whether the two preparations differ chemically, as the literature would predict for EVs vs nanovesicles."),
);

// ------------------------------------------------------------------ 2 experimental
body.push(new Paragraph({ children: [new PageBreak()] }), H1("2. Experimental conditions"));
body.push(P("The samples were received from another laboratory; the isolation protocol is unknown. All acquisition " +
  "parameters below were read from the data file names; the software is LabSpec 6 (Horiba, *.l6s files). Codes whose " +
  "meaning is not stated in the files are marked as interpretations."));
body.push(Tbl(["Parameter", "CaF_{2} (spontaneous Raman)", "AgSiNW (SERS)"], [
  ["Samples / spectra", "EV Acacia: s1, s4, s5, s8, s9, s10 (6 spectra). EV nano Acacia: s1 (1 spectrum)",
    "EV Acacia: maps m1–m5 (5 × 20 = 100 spectra) + single S1, S2. EV nano Acacia: maps m1, m6a (40 spectra) + single s3, s4, s4a"],
  ["Deposition", "Drop-cast (\"dropCaF2\")", "1 µL drop, measured at the drop centre; EV nano Acacia 20 ng"],
  ["Laser", "532 nm", "532 nm"],
  ["Grating", "1800 gr/mm", "1800 gr/mm"],
  ["Objective", "100×", "50× long-working-distance (\"50XLF\")"],
  ["Integration × accumulations", "60 s × 4", "1 s × 4"],
  ["Filter / power code", "\"100\" (interpreted: 100 % laser power)", "\"10\" (interpreted: 10 % laser power)"],
  ["\"BC\" code", "BC50", "BC200 (EV nano Acacia s3, s4: BC100)"],
  ["Spectral range", "~200–3200 cm^-1^", "200–3200 cm^-1^ (S1: 400–2000)"],
], [2000, 3513, 3513]));
body.push(Caption("Table 2. Measurement conditions taken from the file names. If \"100\"/\"10\" are laser-power percentages, " +
  "each AgSiNW spectrum collects about 60 s × 100 % / (1 s × 10 %) = 600 times fewer photon·seconds than a CaF_{2} spectrum. " +
  "The 50× objective also has a lower collection efficiency than the 100× objective."));

body.push(H2("2.1 Data processing"));
body.push(
  B("All analysis was done on **raw counts**, with no smoothing, baseline correction or normalisation of the plotted spectra."),
  B("Noise σ = 1.4826·MAD(first difference)/√2 of each spectrum. Peaks: local maxima with prominence ≥ 5σ and FWHM ≥ 4.5 cm^-1^; narrow maxima were accepted only with SNR ≥ 7 and are flagged."),
  B("Two independent re-checks were applied to every peak. (1) The raw maximum within ±6 cm^-1^ must be the same data point and stand ≥ 3σ above a line drawn between the flank minima. (2) A repeated search with different settings (4σ, ±50 cm^-1^ window) must find the peak again within ±2 cm^-1^."),
  B("On AgSiNW a noise-calibrated search was also used. A band had to exceed the 99th percentile that pure noise reaches in a band-free region (600–720 cm^-1^) of the same spectrum. Spike-like maxima (cosmic rays) were kept only if reproduced in another spectrum."),
  B("Bands in two spectra were counted as the same band if within ±8 cm^-1^. Assignments follow standard biological Raman tables [16–19]. Si bands follow [20] and carbon D/G bands follow [21–24]."),
);

// ------------------------------------------------------------------ 3 results
body.push(new Paragraph({ children: [new PageBreak()] }), H1("3. Results"));
body.push(H2("3.1 EV Acacia: CaF₂ vs AgSiNW"));
body.push(P("On CaF_{2} all five replicate spectra (s4–s10) passed the 1456 cm^-1^ band in 10 of 10 detection checks " +
  "(SNR 18–38, position 1454–1457 cm^-1^). The ~1728 cm^-1^ C=O band passed 9–10 checks (SNR 10–20, 1726–1730 cm^-1^). " +
  "Spectrum s1 was chosen for the slides (Figure 1, Table 3). Of the two AgSiNW single spectra, S2 reproduced the most CaF_{2} " +
  "bands (2 of 7) and was chosen."));
body.push(Fig("EV_Acacia_PPT_Compare/EV_Acacia_CaF2_vs_AgSiNW_400-1800.png", 3999, 2250));
body.push(Caption("Figure 1. EV Acacia on CaF_{2} (s1, top) and on AgSiNW (S2, bottom), raw counts, 400–1800 cm^-1^. " +
  "Green dashed lines mark bands present on both substrates (±8 cm^-1^)."));
body.push(Tbl(["CaF_{2} (cm^-1^)", "AgSiNW (cm^-1^)", "Status", "Assignment"], [
  ["–", "518.8", "only AgSiNW", "Si optical phonon (substrate)"],
  ["532.5", "–", "only CaF_{2}", "unassigned"],
  ["–", "831.3", "only AgSiNW", "Tyr Fermi doublet / O–P–O stretch"],
  ["848.6", "–", "only CaF_{2}", "Tyr ring / C–C (proline), polysaccharide C–O–C"],
  ["–", "1001.6", "only AgSiNW", "Phe ring breathing"],
  ["1118.6", "–", "only CaF_{2}", "C–C stretch (trans lipid) / C–N (protein)"],
  ["–", "1166.4", "only AgSiNW", "Tyr C–H bend"],
  ["1255.5", "–", "only CaF_{2}", "Amide III"],
  ["–", "1380.9", "only AgSiNW", "COO^-^ sym. stretch / CH_{3} bend / carbon D band"],
  ["1456.8", "–", "only CaF_{2}", "CH_{2}/CH_{3} deformation (lipid, protein)"],
  ["–", "1519.8", "only AgSiNW", "Carotenoid-type C=C stretch"],
  ["–", "1580.1", "only AgSiNW", "COO^-^ asym. stretch / carbon G band / Phe"],
  ["1605.4", "1607.1", "in both", "Phe/Tyr ring C=C (on AgSiNW overlaps G band)"],
  ["–", "1632.9", "only AgSiNW", "unassigned"],
  ["1729.4", "1722.3", "in both", "C=O stretch (ester, lipids)"],
], [1500, 1500, 1500, 4526], { fill: statusFill }));
body.push(Caption("Table 3. Confirmed bands of EV Acacia, 400–1800 cm^-1^ (raw data). Green rows: band found on both substrates."));

body.push(P("**Search of the 100 AgSiNW map spectra for 1456 and 1726 cm^-1^.** The 1456 cm^-1^ band reached SNR ≥ 5 in 7 spectra " +
  "and the 1726 cm^-1^ band in 6 spectra, but **no spectrum showed both at SNR ≥ 5**. The best pixel (m5, spectrum 12) reached " +
  "only SNR 5.5 and 4.3. On CaF_{2} the same bands have SNR 18–38 and 10–20. The C=O band appears at 1722 cm^-1^ in S2, " +
  "but without the 1456 cm^-1^ band."));
body.push(P("**Carbon D/G bands.** Fits of the 1200–1700 cm^-1^ region of the SERS spectra (Table 4) show a broad D band " +
  "(1360–1378 cm^-1^, FWHM 89–227 cm^-1^), a G band (1605–1608 cm^-1^) and a D3 band (~1545 cm^-1^). This is the typical signature of " +
  "disordered or amorphous carbon [21–24]. None of these bands appears on CaF_{2}."));
body.push(Tbl(["Spectrum", "D (cm^-1^)", "D FWHM", "G (cm^-1^)", "G FWHM", "D3 (cm^-1^)", "I_{D}/I_{G}", "R^2^"], [
  ["EV Acacia, mean of 100 map spectra", "1360.3", "227", "1604.9", "64", "1543.8", "0.73", "0.96"],
  ["EV Acacia S1", "1377.7", "89", "1608.3", "75", "1550.0", "0.26", "0.94"],
  ["EV Acacia S2", "1366.6", "96", "1607.4", "50", "1547.6", "0.86", "0.87"],
  ["EV nano Acacia s3", "1337.3", "–", "1620.0", "–", "–", "1.52", "0.75"],
  ["EV nano Acacia s4", "1385.1", "–", "1608.2", "–", "–", "0.60", "0.91"],
  ["EV nano Acacia s4a", "1384.4", "–", "1614.4", "–", "–", "0.61", "0.94"],
  ["EV nano Acacia, mean map m1", "1381.5", "–", "1612.0", "–", "–", "0.60", "0.95"],
], [2626, 900, 800, 900, 800, 900, 1100, 1000]));
body.push(Caption("Table 4. Carbon D/G band fits of the SERS spectra (Lorentzian D, G and Gaussian D3 on a linear background)."));
body.push(Fig("EV_Acacia_DG_Bands/DG_bands_plot.png", 1980, 2059, 430));
body.push(Caption("Figure 2. D/G band fits for EV Acacia on AgSiNW compared with the CaF_{2} spectra."));

body.push(H2("3.2 EV nano Acacia: CaF₂ vs AgSiNW"));
body.push(P("Only one CaF_{2} spectrum (s1) of EV nano Acacia was available. Of the three AgSiNW single spectra, s4 matched " +
  "the most CaF_{2} bands, but this was only 1 of 10 (Figure 3, Table 5). The 1456 and 1726 cm^-1^ bands were not detected in any " +
  "EV nano Acacia spectrum on AgSiNW (maximum SNR 3.4 for 1456 and 3.7 for 1726 cm^-1^ over all 43 spectra)."));
body.push(Fig("EV_nano_Acacia_PPT_Compare/EV_nano_Acacia_CaF2_vs_AgSiNW_400-1800.png", 3999, 2250));
body.push(Caption("Figure 3. EV nano Acacia on CaF_{2} (s1, top) and on AgSiNW (s4, bottom), raw counts, 400–1800 cm^-1^."));
body.push(Tbl(["CaF_{2} (cm^-1^)", "AgSiNW (cm^-1^)", "Status", "Assignment"], [
  ["435.8", "–", "only CaF_{2}", "unassigned (skeletal deformation region)"],
  ["459.7", "–", "only CaF_{2}", "unassigned (skeletal deformation region)"],
  ["–", "519.3", "only AgSiNW", "Si optical phonon (substrate)"],
  ["851.4", "–", "only CaF_{2}", "Tyr ring / C–C (proline), polysaccharide C–O–C"],
  ["899.6", "–", "only CaF_{2}", "C–C / C–O–C (saccharide)"],
  ["–", "926.5", "only AgSiNW", "C–C stretch, protein backbone"],
  ["–", "955.5", "only AgSiNW", "Si 2nd-order phonon (substrate)"],
  ["–", "1001.6", "only AgSiNW", "Phe ring breathing"],
  ["1096.4", "–", "only CaF_{2}", "C–O/C–C (carbohydrate), PO_{2}^-^ sym., gauche C–C (strongest band, SNR 46)"],
  ["1118.6", "–", "only CaF_{2}", "C–C stretch (trans lipid) / C–N (protein)"],
  ["–", "1244.4", "only AgSiNW", "Amide III"],
  ["1291.3", "–", "only CaF_{2}", "CH_{2} twist (lipid)"],
  ["1339.0", "–", "only CaF_{2}", "CH_{3}CH_{2} wagging (protein, nucleic acid)"],
  ["–", "1387.4", "only AgSiNW", "COO^-^ sym. stretch / carbon D band"],
  ["1465.4", "–", "only CaF_{2}", "CH_{2}/CH_{3} deformation"],
  ["–", "1574.1", "only AgSiNW", "Amide II / Trp / carbon G band region"],
  ["1603.7", "1606.7", "in both", "Phe/Tyr ring C=C (on AgSiNW overlaps G band)"],
  ["–", "1617.2", "only AgSiNW", "Tyr/Trp ring C=C"],
], [1500, 1500, 1500, 4526], { fill: statusFill }));
body.push(Caption("Table 5. Confirmed bands of EV nano Acacia, 400–1800 cm^-1^ (raw data)."));

body.push(H2("3.3 EV Acacia vs EV nano Acacia on CaF₂"));
body.push(P("This is the most reliable comparison, because both samples were measured with identical settings on a " +
  "low-background substrate. Bands were searched in all spectra of each sample (six EV Acacia, one EV nano Acacia) and placed at the " +
  "strongest reproduced maximum. For this reason, some positions differ by a few cm^-1^ from the single-spectrum lists in Tables 3 and 5."));
body.push(Fig("EV_Acacia_vs_Nano_CaF2/EV_Acacia_vs_nano_CaF2_400-1800.png", 3999, 2250));
body.push(Caption("Figure 4. EV Acacia (top) and EV nano Acacia (bottom) on CaF_{2}, raw counts, 400–1800 cm^-1^."));
body.push(Tbl(["EV Acacia (cm^-1^)", "EV nano (cm^-1^)", "Status", "Assignment"], [
  ["–", "435.8", "only nano", "unassigned"],
  ["–", "459.7", "only nano", "unassigned"],
  ["482.9", "–", "only Acacia", "unassigned"],
  ["848.6", "851.4", "in both", "Tyr ring / C–C (proline), polysaccharide C–O–C"],
  ["–", "899.6", "only nano", "C–C / C–O–C (saccharide)"],
  ["927.5", "–", "only Acacia", "C–C stretch, protein backbone"],
  ["–", "978.6", "only nano", "unassigned"],
  ["–", "1002.1", "only nano", "Phe ring breathing"],
  ["1076.9", "–", "only Acacia", "PO_{2}^-^ sym. / gauche C–C / C–O"],
  ["–", "1096.4", "only nano", "C–O/C–C (carbohydrate), PO_{2}^-^ sym. (strongest nano band)"],
  ["1115.0", "1118.6", "in both", "C–C stretch (trans lipid) / C–N (protein)"],
  ["1244.8", "1258.1", "only one each (Δ13)", "Amide III"],
  ["1305.5", "–", "only Acacia", "CH_{2} twist (lipid)"],
  ["1341.6", "1339.0", "in both", "CH_{3}CH_{2} wagging (protein, nucleic acid)"],
  ["1456.8", "1465.4", "only one each (Δ8.6)", "CH_{2}/CH_{3} deformation"],
  ["1559.2", "–", "only Acacia", "Amide II / Trp"],
  ["1605.4", "1603.7", "in both", "Phe/Tyr ring C=C"],
  ["1729.4", "–", "only Acacia", "C=O stretch (ester, lipids)"],
], [1500, 1500, 1700, 4326], { fill: statusFill }));
body.push(Caption("Table 6. EV Acacia vs EV nano Acacia on CaF_{2}. Only four bands are shared."));
body.push(P("Whole-spectrum correlations with the EV Acacia mean spectrum confirm the difference. The EV Acacia replicates give " +
  "r = 0.96–0.99 in the fingerprint region and ≥ 0.999 in the CH region (2750–3100 cm^-1^). EV nano Acacia gives r = 0.77 and 0.89. " +
  "In the CH region the EV Acacia maximum is at ~2937 cm^-1^ (CH_{3}/CH_{2} asymmetric, protein and lipid). The EV nano Acacia maximum is at " +
  "~2891 cm^-1^ (CH_{2} symmetric stretch or C–H of carbohydrate)."));

body.push(H2("3.4 EV Acacia vs EV nano Acacia on AgSiNW"));
body.push(Fig("EV_Acacia_vs_Nano_AgSiNW/EV_Acacia_vs_nano_AgSiNW_400-1800.png", 3999, 2250));
body.push(Caption("Figure 5. EV Acacia (S2, top) and EV nano Acacia (s4, bottom) on AgSiNW, raw counts, 400–1800 cm^-1^."));
body.push(Tbl(["EV Acacia (cm^-1^)", "EV nano (cm^-1^)", "Status", "Assignment"], [
  ["518.8", "519.3", "in both", "Si optical phonon (substrate)"],
  ["831.3", "–", "only Acacia", "Tyr Fermi doublet / O–P–O"],
  ["–", "926.5", "only nano", "C–C stretch, protein backbone"],
  ["–", "955.5", "only nano", "Si 2nd-order phonon (substrate)"],
  ["1001.6", "1001.6", "in both", "Phe ring breathing"],
  ["1166.4", "–", "only Acacia", "Tyr C–H bend"],
  ["–", "1244.4", "only nano", "Amide III"],
  ["1380.9", "1387.4", "in both", "COO^-^ sym. stretch / carbon D band"],
  ["1519.8", "–", "only Acacia", "Carotenoid-type C=C"],
  ["1580.1", "1574.1", "in both", "COO^-^ asym. / carbon G band / Phe"],
  ["1607.1", "1606.7", "in both", "Phe/Tyr ring C=C / carbon G band"],
  ["–", "1617.2", "only nano", "Tyr/Trp ring C=C"],
  ["1632.9", "–", "only Acacia", "unassigned"],
  ["1722.3", "–", "only Acacia", "C=O stretch (ester, lipids)"],
], [1500, 1500, 1500, 4526], { fill: statusFill }));
body.push(Caption("Table 7. EV Acacia vs EV nano Acacia on AgSiNW. The shared bands are the substrate, Phe and the carboxylate/carbon region."));
body.push(P("On AgSiNW the two samples look more alike than on CaF_{2}. The shared bands (Si, Phe, COO^-^/D/G) are mostly substrate, " +
  "aromatic amino acids adsorbed on silver, and photo-generated carbon. The only lipid marker, C=O at 1722 cm^-1^, appears in " +
  "EV Acacia and not in EV nano Acacia. This agrees in direction with the CaF_{2} result."));

// ------------------------------------------------------------------ 4 discussion
body.push(new Paragraph({ children: [new PageBreak()] }), H1("4. Interpretation"));
body.push(H2("4.1 The two preparations are chemically different"));
body.push(P("The CaF_{2} data show that **EV Acacia** is rich in ester-containing lipids (C=O 1729, CH_{2}/CH_{3} 1456, CH_{2} twist 1305, " +
  "CH 2937 cm^-1^) and also contains protein (amide III 1245–1255, amide II/Trp 1559, Phe/Tyr 1605, C–C 927 cm^-1^) [16–18]. This is " +
  "what a membrane vesicle with phospholipid bilayer and associated proteins is expected to show. The narrow band positions and " +
  "high replicate correlation also show that the preparation is homogeneous."));
body.push(P("**EV nano Acacia** lacks the ester C=O band. Its strongest bands are at 1096 and 900 cm^-1^, which are typical of C–O/C–C and " +
  "glycosidic C–O–C vibrations of carbohydrates [19], and at 2891 cm^-1^ in the CH region. Plant nanovesicles made by homogenisation " +
  "carry co-isolated biomass material [1–3]. Acacia tissue and exudates are very rich in polysaccharide gums and polyphenols [10,11]. " +
  "A carbohydrate-dominated spectrum without a strong ester band is therefore **consistent with** a homogenate-derived nanovesicle " +
  "preparation. The 1096 cm^-1^ band can also contain a phosphate (PO_{2}^-^) contribution. This interpretation is a hypothesis: the " +
  "isolation methods are unknown and only one CaF_{2} spectrum of EV nano Acacia exists."));

body.push(H2("4.2 Why the CaF₂ bands (1456, 1726 cm⁻¹) are missing on AgSiNW"));
body.push(
  B("**SERS is selective.** Enhancement is strongest for molecules within a few nanometres of a silver hot spot. It also depends on " +
    "their orientation relative to the surface (surface selection rules) [15,25,26]. Aromatic side chains (Phe, Tyr) and carboxylates " +
    "adsorb on silver and dominate. The ester groups and alkyl chains inside a bilayer are further away and are not enhanced."),
  B("**Much smaller signal budget.** 1 s at a 10 % power setting with a 50× LF objective, against 60 s at 100 % with 100×, gives about " +
    "600 times fewer photon·seconds before collection-efficiency losses. Bands at SNR 20–38 on CaF_{2} fall to the noise level unless they " +
    "are strongly enhanced."),
  B("**Photo-generated carbon.** Laser heating at silver hot spots degrades organic material to amorphous carbon. This produces broad " +
    "D (~1360–1385) and G (~1580–1610 cm^-1^) bands [21–24], which were found in all SERS spectra (Table 4). They overlap the 1380–1610 cm^-1^ region " +
    "and raise the background near 1456 cm^-1^."),
  B("**Substrate and dilution.** Si phonons at 519 and ~955 cm^-1^ [20] and a very small analyte amount (1 µL; 20 ng of EV nano Acacia) reduce the " +
    "fraction of the spectrum carrying vesicle information."),
);
body.push(P("The missing bands are therefore a property of the SERS measurement, not proof that the vesicles lack lipids. For compositional " +
  "comparison the CaF_{2} spectra should be used. SERS is useful as a sensitivity test and for protein/aromatic markers."));

body.push(H2("4.3 What this means for possible uses"));
body.push(P("If EV Acacia is an apoplastic or secreted EV preparation, its lipid- and protein-rich membrane signature suits studies of plant " +
  "vesicle biology and use as natural carriers, where membrane integrity matters [1,4,5]. If EV nano Acacia is a homogenate-derived " +
  "nanovesicle preparation, its carbohydrate and polyphenol load fits the application area where plant nanovesicles are most studied. " +
  "That area is oral, gut and anti-inflammatory use [6–8], where Acacia's own anti-inflammatory and antioxidant compounds [10,11] could add " +
  "to the effect. Both statements are conditional on the isolation methods being confirmed."));

// ------------------------------------------------------------------ 5 limitations / recommendations
body.push(H1("5. Limitations"));
body.push(
  B("The isolation method, plant species and organ, particle concentration (except 20 ng for EV nano Acacia on AgSiNW) and storage history are unknown."),
  B("EV nano Acacia on CaF_{2} is a single spectrum, so its reproducibility cannot be tested."),
  B("No blank-substrate spectra (bare CaF_{2}, bare AgSiNW) or buffer controls were provided."),
  B("AgSiNW settings differ between spectra (BC100 vs BC200), and power/filter codes were interpreted from file names."),
  B("No particle size, morphology or marker data (NTA, TEM, protein markers) were available, as MISEV2023 recommends [27]."),
  B("Band assignments are tentative; several bands (e.g. 1096, 1380, 1605 cm^-1^) have overlapping contributions."),
);
body.push(H1("6. Recommendations"));
body.push(
  B("Ask the provider lab for the isolation protocol (apoplastic wash or homogenisation; ultracentrifugation, SEC or density gradient), " +
    "species, organ and particle concentration. Report them following MISEV2023 [27] and Pinedo et al. [2]."),
  B("Add NTA or DLS (size, concentration), TEM (morphology) and protein quantification for both samples."),
  B("Record more CaF_{2} spectra of EV nano Acacia (≥ 5 positions) and bare-substrate and buffer blanks."),
  B("Measure reference compounds on CaF_{2} under the same conditions: phosphatidylcholine, gum arabic (Acacia senegal gum) and tannic acid. " +
    "These test the lipid vs polysaccharide/polyphenol assignment directly."),
  B("For SERS, use lower laser power with longer integration to avoid carbonisation (check for the D/G bands), " +
    "keep one BC setting, and consider aggregating the vesicles on the substrate or using a spacer-free Ag colloid."),
);

// ------------------------------------------------------------------ references
body.push(new Paragraph({ children: [new PageBreak()] }), H1("References"));
[
  "Alsaid et al. (2026). Plant-derived extracellular vesicles and nanoparticles: origins, functions, and applications. Frontiers in Bioengineering and Biotechnology. doi:10.3389/fbioe.2026.1758558",
  "Pinedo M., de la Canal L., de Marcos Lousa C. (2021). A call for Rigor and standardization in plant extracellular vesicle research. Journal of Extracellular Vesicles 10:e12048. doi:10.1002/jev2.12048",
  "Woith E., Guerriero G., Hausman J.-F., et al. (2021). Plant extracellular vesicles and nanovesicles: focus on secondary metabolites, proteins and lipids with perspectives on their potential and sources. International Journal of Molecular Sciences 22:3719. doi:10.3390/ijms22073719",
  "Rutter B.D., Innes R.W. (2017). Extracellular vesicles isolated from the leaf apoplast carry stress-response proteins. Plant Physiology 173:728–741. doi:10.1104/pp.16.01253",
  "Kocholatá M., Malý J., Kříženecká S., Janoušková O. (2024). Diversity of extracellular vesicles derived from calli, cell culture and apoplastic fluid of tobacco. Scientific Reports 14. doi:10.1038/s41598-024-81940-8",
  "Mu J., Zhuang X., Wang Q., et al. (2014). Interspecies communication between plant and mouse gut host cells through edible plant derived exosome-like nanoparticles. Molecular Nutrition & Food Research 58:1561–1573. doi:10.1002/mnfr.201300729",
  "Zhang M., Viennois E., Prasad M., et al. (2016). Edible ginger-derived nanoparticles: a novel therapeutic approach for the prevention and treatment of inflammatory bowel disease and colitis-associated cancer. Biomaterials 101:321–340. doi:10.1016/j.biomaterials.2016.06.018",
  "Chen X., Liu B., Li X., et al. (2021). Identification of anti-inflammatory vesicle-like nanoparticles in honey. Journal of Extracellular Vesicles 10:e12069. doi:10.1002/jev2.12069",
  "Lian M.Q., Chng W.H., Liang J., et al. (2022). Plant-derived extracellular vesicles: recent advancements and current challenges on their use for biomedical applications. Journal of Extracellular Vesicles 11:e12283. doi:10.1002/jev2.12283",
  "Rather L.J., Shahid-ul-Islam, Mohammad F. (2015). Acacia nilotica (L.): a review of its traditional uses, phytochemistry, and pharmacology. Sustainable Chemistry and Pharmacy 2:12–30. doi:10.1016/j.scp.2015.08.002",
  "Hafez L.O., et al. (2024). The Acacia (Vachellia nilotica (L.) P.J.H. Hurter & Mabb.): traditional uses and recent advances on its pharmacological attributes and potential activities. PMID 39770900; PMCID PMC11678605",
  "Gualerzi A., Niada S., Giannasi C., et al. (2017). Raman spectroscopy uncovers biochemical tissue-related features of extracellular vesicles from mesenchymal stromal cells. Scientific Reports 7:9820. doi:10.1038/s41598-017-10448-1",
  "Stremersch S., Marro M., Pinchasik B.-E., et al. (2016). Identification of individual exosome-like vesicles by surface enhanced Raman spectroscopy. Small 12:3292–3301. doi:10.1002/smll.201600393",
  "Kerr L.T., Byrne H.J., Hennelly B.M. (2015). Optimal choice of sample substrate and laser wavelength for Raman spectroscopic analysis of biological specimen. Analytical Methods 7:5041–5052. doi:10.1039/C5AY00327J",
  "Otto A. (2002). What is observed in single molecule SERS, and why? Journal of Raman Spectroscopy 33:593–598. doi:10.1002/jrs.879",
  "Movasaghi Z., Rehman S., Rehman I.U. (2007). Raman spectroscopy of biological tissues. Applied Spectroscopy Reviews 42:493–541. doi:10.1080/05704920701551530",
  "Czamara K., Majzner K., Pacia M.Z., et al. (2015). Raman spectroscopy of lipids: a review. Journal of Raman Spectroscopy 46:4–20. doi:10.1002/jrs.4607",
  "Rygula A., Majzner K., Marzec K.M., et al. (2013). Raman spectroscopy of proteins: a review. Journal of Raman Spectroscopy 44:1061–1076. doi:10.1002/jrs.4335",
  "Wiercigroch E., Szafraniec E., Czamara K., et al. (2017). Raman and infrared spectroscopy of carbohydrates: a review. Spectrochimica Acta A 185:317–335. doi:10.1016/j.saa.2017.05.045",
  "Temple P.A., Hathaway C.E. (1973). Multiphonon Raman spectrum of silicon. Physical Review B 7:3685–3697. doi:10.1103/PhysRevB.7.3685",
  "Tuinstra F., Koenig J.L. (1970). Raman spectrum of graphite. Journal of Chemical Physics 53:1126–1130. doi:10.1063/1.1674108",
  "Ferrari A.C., Robertson J. (2000). Interpretation of Raman spectra of disordered and amorphous carbon. Physical Review B 61:14095–14107. doi:10.1103/PhysRevB.61.14095",
  "Sadezky A., Muckenhuber H., Grothe H., Niessner R., Pöschl U. (2005). Raman microspectroscopy of soot and related carbonaceous materials: spectral analysis and structural information. Carbon 43:1731–1742. doi:10.1016/j.carbon.2005.02.018",
  "Kudelski A., Pettinger B. (2000). SERS on carbon chain segments: monitoring locally surface chemistry. Chemical Physics Letters 321:356–362. doi:10.1016/S0009-2614(00)00370-8",
  "Moskovits M. (1982). Surface selection rules. Journal of Chemical Physics 77:4408–4416. doi:10.1063/1.444442",
  "Moskovits M. (1985). Surface-enhanced spectroscopy. Reviews of Modern Physics 57:783–826. doi:10.1103/RevModPhys.57.783",
  "Welsh J.A., Goberdhan D.C.I., O'Driscoll L., et al. (2024). Minimal information for studies of extracellular vesicles (MISEV2023): from basic to advanced approaches. Journal of Extracellular Vesicles 13:e12404. doi:10.1002/jev2.12404",
].forEach((r) => body.push(N(r)));

// ------------------------------------------------------------------ document
const doc = new Document({
  creator: "Raman analysis",
  title: "Raman and SERS characterisation of EV Acacia and EV nano Acacia",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: FONT, color: "1F4E79" }, paragraph: { spacing: { before: 300, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT, color: "2E75B6" }, paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "refs", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "[%1]", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 540 } } } }] },
  ] },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "808080" })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync("EV_Acacia_Report.docx", b); console.log("written"); });

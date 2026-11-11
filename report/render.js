// Renders the block list from content.js to .docx (docx-js) and .pdf (HTML via headless Chromium).
// Inline markup in strings: **bold**, *italic*, `code`, ^sup^, ~sub~.
const fs = require("fs");
const path = require("path");
const D = require("docx");

const W = 9026; // A4 content width in DXA with 1-inch margins

function parseInline(s) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\^[^^]+\^|~[^~]+~)/g;
  let last = 0, m;
  s = String(s);
  while ((m = re.exec(s))) {
    if (m.index > last) out.push({ text: s.slice(last, m.index) });
    const t = m[0];
    if (t.startsWith("**")) out.push({ text: t.slice(2, -2), bold: true });
    else if (t.startsWith("*")) out.push({ text: t.slice(1, -1), italics: true });
    else if (t.startsWith("`")) out.push({ text: t.slice(1, -1), code: true });
    else if (t.startsWith("^")) out.push({ text: t.slice(1, -1), sup: true });
    else out.push({ text: t.slice(1, -1), sub: true });
    last = m.index + t.length;
  }
  if (last < s.length) out.push({ text: s.slice(last) });
  return out;
}
const esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

// ---------- DOCX ----------
function runs(s, base = {}) {
  return parseInline(s).map((r) => new D.TextRun({
    text: r.text, bold: r.bold || base.bold, italics: r.italics || base.italics,
    font: r.code ? "Consolas" : base.font, size: r.code ? (base.size || 22) - 2 : base.size,
    superScript: r.sup, subScript: r.sub, color: base.color,
  }));
}
const border = { style: D.BorderStyle.SINGLE, size: 4, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };

function docxBlocks(blocks) {
  const out = [];
  for (const b of blocks) {
    switch (b.t) {
      case "title":
        out.push(new D.Paragraph({ alignment: D.AlignmentType.CENTER, spacing: { after: 80 }, children: runs(b.text, { bold: true, size: 30 }) }));
        break;
      case "center":
        out.push(new D.Paragraph({ alignment: D.AlignmentType.CENTER, spacing: { after: 40 }, children: runs(b.text) }));
        break;
      case "h1":
        out.push(new D.Paragraph({ heading: D.HeadingLevel.HEADING_1, children: runs(b.text) }));
        break;
      case "h2":
        out.push(new D.Paragraph({ heading: D.HeadingLevel.HEADING_2, children: runs(b.text) }));
        break;
      case "h3":
        out.push(new D.Paragraph({ heading: D.HeadingLevel.HEADING_3, children: runs(b.text) }));
        break;
      case "p":
        out.push(new D.Paragraph({ alignment: D.AlignmentType.JUSTIFIED, spacing: { after: 120 }, children: runs(b.text) }));
        break;
      case "list":
        b.items.forEach((it) => out.push(new D.Paragraph({
          numbering: { reference: b.ordered ? "num" + (b.ref || "a") : "bullets", level: 0 },
          spacing: { after: 60 }, children: runs(it) })));
        break;
      case "steps":
        b.items.forEach(([k, v]) => out.push(new D.Paragraph({
          indent: { left: 1080, hanging: 1080 }, spacing: { after: 80 },
          children: [...runs(k + "\t", { bold: true }), ...runs(v)],
          tabStops: [{ type: D.TabStopType.LEFT, position: 1080 }] })));
        break;
      case "eq":
        out.push(new D.Paragraph({
          tabStops: [{ type: D.TabStopType.CENTER, position: W / 2 }, { type: D.TabStopType.RIGHT, position: W }],
          spacing: { before: 60, after: 60 },
          children: [new D.TextRun("\t"), ...runs(b.text, { font: "Cambria Math" }), new D.TextRun("\t(" + b.n + ")")] }));
        break;
      case "code": {
        const lines = b.text.replace(/\t/g, "    ").split("\n");
        lines.forEach((ln, i) => out.push(new D.Paragraph({
          spacing: { after: 0, line: 228 }, shading: { type: D.ShadingType.CLEAR, fill: "F4F4F4" },
          keepLines: true,
          children: [new D.TextRun({ text: ln.length ? ln : " ", font: "Consolas", size: b.size || 15 })] })));
        out.push(new D.Paragraph({ spacing: { after: 120 }, children: [] }));
        break;
      }
      case "img": {
        const data = fs.readFileSync(b.src);
        const [pw, ph] = pngSize(data);
        const wpx = Math.round(b.w * 96), hpx = Math.round(wpx * ph / pw);
        out.push(new D.Paragraph({ alignment: D.AlignmentType.CENTER, keepNext: true,
          children: [new D.ImageRun({ type: "png", data, transformation: { width: wpx, height: hpx } })] }));
        out.push(caption(b.caption));
        break;
      }
      case "table": {
        const fr = b.cols, tot = fr.reduce((a, c) => a + c, 0);
        const widths = fr.map((c) => Math.floor((W * c) / tot));
        widths[widths.length - 1] += W - widths.reduce((a, c) => a + c, 0);
        const sz = b.size || 17;
        const mk = (cells, head) => new D.TableRow({ tableHeader: head, cantSplit: true, children: cells.map((c, i) => new D.TableCell({
          borders, width: { size: widths[i], type: D.WidthType.DXA },
          margins: { top: 40, bottom: 40, left: 80, right: 80 },
          shading: head ? { type: D.ShadingType.CLEAR, fill: "E3EAF3" } : (b.shade && b.shade(cells) ? { type: D.ShadingType.CLEAR, fill: "F6F6F6" } : undefined),
          children: String(c).split("\n").map((ln) => new D.Paragraph({
            alignment: (b.align && b.align[i] === "r") ? D.AlignmentType.RIGHT : D.AlignmentType.LEFT,
            children: runs(ln, { size: sz, bold: head }) })) })) });
        out.push(new D.Table({ width: { size: W, type: D.WidthType.DXA }, columnWidths: widths,
          rows: [mk(b.head, true), ...b.rows.map((r) => mk(r, false))] }));
        out.push(caption(b.caption));
        break;
      }
      case "pagebreak":
        out.push(new D.Paragraph({ children: [new D.PageBreak()] }));
        break;
      case "ref":
        out.push(new D.Paragraph({ indent: { left: 567, hanging: 567 }, spacing: { after: 80 }, children: runs(b.text, { size: 20 }) }));
        break;
      default: throw new Error("unknown block " + b.t);
    }
  }
  return out;
}
function caption(text) {
  return new D.Paragraph({ alignment: D.AlignmentType.CENTER, spacing: { before: 80, after: 200 },
    children: runs(text, { italics: true, size: 18 }) });
}
function pngSize(buf) { return [buf.readUInt32BE(16), buf.readUInt32BE(20)]; }

async function writeDocx(blocks, file) {
  const refs = [...new Set(blocks.filter((b) => b.t === "list" && b.ordered).map((b) => b.ref || "a"))];
  const doc = new D.Document({
    creator: "Ayauzhan Baurzhanova, Aiat Mergenbay",
    title: "Assignment 3 – Methodology and Core Practical Implementation",
    styles: {
      default: { document: { run: { font: "Calibri", size: 22 } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 28, bold: true, color: "1F3864" }, paragraph: { spacing: { before: 300, after: 120 }, keepNext: true, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 24, bold: true, color: "1F3864" }, paragraph: { spacing: { before: 220, after: 100 }, keepNext: true, outlineLevel: 1 } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 22, bold: true, italics: true }, paragraph: { spacing: { before: 160, after: 80 }, keepNext: true, outlineLevel: 2 } },
      ],
    },
    numbering: { config: [
      { reference: "bullets", levels: [{ level: 0, format: D.LevelFormat.BULLET, text: "•", alignment: D.AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 567, hanging: 283 } } } }] },
      ...refs.map((r) => ({ reference: "num" + r, levels: [{ level: 0, format: D.LevelFormat.DECIMAL, text: "%1.", alignment: D.AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 567, hanging: 340 } } } }] })),
    ] },
    sections: [{
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
      footers: { default: new D.Footer({ children: [new D.Paragraph({ alignment: D.AlignmentType.CENTER,
        children: [new D.TextRun({ children: [D.PageNumber.CURRENT], size: 18 })] })] }) },
      children: docxBlocks(blocks),
    }],
  });
  fs.writeFileSync(file, await D.Packer.toBuffer(doc));
}

// ---------- HTML / PDF ----------
function html(s) {
  return parseInline(s).map((r) => {
    let t = esc(r.text);
    if (r.code) t = `<code>${t}</code>`;
    if (r.bold) t = `<b>${t}</b>`;
    if (r.italics) t = `<i>${t}</i>`;
    if (r.sup) t = `<sup>${t}</sup>`;
    if (r.sub) t = `<sub>${t}</sub>`;
    return t;
  }).join("");
}
function htmlBlocks(blocks) {
  let o = "";
  for (const b of blocks) {
    switch (b.t) {
      case "title": o += `<div class="title">${html(b.text)}</div>`; break;
      case "center": o += `<div class="center">${html(b.text)}</div>`; break;
      case "h1": o += `<h1>${html(b.text)}</h1>`; break;
      case "h2": o += `<h2>${html(b.text)}</h2>`; break;
      case "h3": o += `<h3>${html(b.text)}</h3>`; break;
      case "p": o += `<p>${html(b.text)}</p>`; break;
      case "list": { const tag = b.ordered ? "ol" : "ul"; o += `<${tag}>${b.items.map((i) => `<li>${html(i)}</li>`).join("")}</${tag}>`; break; }
      case "steps": o += `<div class="steps">${b.items.map(([k, v]) => `<div class="step"><b>${html(k)}</b><span>${html(v)}</span></div>`).join("")}</div>`; break;
      case "eq": o += `<div class="eq"><span></span><span class="m">${html(b.text)}</span><span>(${b.n})</span></div>`; break;
      case "code": o += `<pre class="${b.size && b.size < 15 ? "tiny" : ""}">${esc(b.text.replace(/\t/g, "    "))}</pre>`; break;
      case "img": o += `<figure><img src="file://${path.resolve(b.src)}" style="width:${b.w}in"><figcaption>${html(b.caption)}</figcaption></figure>`; break;
      case "table": {
        const tot = b.cols.reduce((a, c) => a + c, 0);
        o += `<table class="${b.size && b.size < 17 ? "small" : ""}"><colgroup>${b.cols.map((c) => `<col style="width:${(100 * c / tot).toFixed(1)}%">`).join("")}</colgroup>`;
        o += `<thead><tr>${b.head.map((h) => `<th>${String(h).split("\n").map(html).join("<br>")}</th>`).join("")}</tr></thead><tbody>`;
        o += b.rows.map((r) => `<tr${b.shade && b.shade(r) ? ' class="sh"' : ""}>${r.map((c, i) => `<td${b.align && b.align[i] === "r" ? ' class="r"' : ""}>${String(c).split("\n").map(html).join("<br>")}</td>`).join("")}</tr>`).join("");
        o += `</tbody></table><div class="cap">${html(b.caption)}</div>`;
        break;
      }
      case "pagebreak": o += `<div class="pb"></div>`; break;
      case "ref": o += `<p class="ref">${html(b.text)}</p>`; break;
    }
  }
  return o;
}
const CSS = `
@page { size: A4; margin: 2.3cm 2.3cm 2.2cm 2.3cm; }
body { font-family: Carlito, Calibri, "Liberation Sans", sans-serif; font-size: 11pt; line-height: 1.38; color: #111; }
.title { text-align: center; font-weight: bold; font-size: 15pt; margin-bottom: 4pt; }
.center { text-align: center; }
h1 { font-size: 14pt; color: #1F3864; margin: 18pt 0 6pt; break-after: avoid; }
h2 { font-size: 12pt; color: #1F3864; margin: 13pt 0 5pt; break-after: avoid; }
h3 { font-size: 11pt; font-style: italic; margin: 10pt 0 4pt; break-after: avoid; }
p { text-align: justify; margin: 0 0 6pt; }
ul, ol { margin: 0 0 6pt; padding-left: 26pt; } li { margin-bottom: 3pt; }
code { font-family: "DejaVu Sans Mono", Consolas, monospace; font-size: 9pt; }
pre { font-family: "DejaVu Sans Mono", Consolas, monospace; font-size: 7.4pt; line-height: 1.3; background: #f4f4f4; padding: 5pt 7pt; white-space: pre-wrap; word-break: break-all; margin: 0 0 8pt; }
pre.tiny { font-size: 6.8pt; }
.steps .step { display: grid; grid-template-columns: 54pt 1fr; margin-bottom: 4pt; text-align: justify; }
.eq { display: grid; grid-template-columns: 1fr auto 1fr; margin: 4pt 0; } .eq span:last-child { text-align: right; } .eq .m { font-family: "DejaVu Serif", Cambria, serif; font-size: 10.5pt; }
figure { margin: 6pt 0 10pt; text-align: center; break-inside: avoid; } figure img { max-width: 100%; }
figcaption, .cap { font-style: italic; font-size: 9pt; text-align: center; margin: 4pt 0 10pt; }
table { width: 100%; border-collapse: collapse; font-size: 8.6pt; line-height: 1.25; }
table.small { font-size: 7.8pt; }
th, td { border: 0.6pt solid #999; padding: 2.5pt 4pt; vertical-align: top; text-align: left; }
th { background: #E3EAF3; } td.r { text-align: right; } tr.sh td { background: #f6f6f6; }
tr { break-inside: avoid; } thead { display: table-header-group; }
.pb { break-after: page; }
.ref { padding-left: 22pt; text-indent: -22pt; font-size: 10pt; text-align: left; margin-bottom: 4pt; }
`;
async function writePdf(blocks, file) {
  const page = `<!doctype html><html><head><meta charset="utf-8"><style>${CSS}</style></head><body>${htmlBlocks(blocks)}</body></html>`;
  const htmlFile = file.replace(/\.pdf$/, ".html");
  fs.writeFileSync(htmlFile, page);
  const puppeteer = require("puppeteer-core");
  const browser = await puppeteer.launch({ executablePath: "/usr/bin/chromium", args: ["--allow-file-access-from-files"] });
  const p = await browser.newPage();
  await p.goto("file://" + path.resolve(htmlFile), { waitUntil: "load" });
  await p.pdf({ path: file, format: "A4", printBackground: true, displayHeaderFooter: true, preferCSSPageSize: true,
    headerTemplate: "<span></span>",
    footerTemplate: '<div style="width:100%;text-align:center;font-size:8pt;font-family:sans-serif;"><span class="pageNumber"></span></div>' });
  await browser.close();
}
module.exports = { writeDocx, writePdf };

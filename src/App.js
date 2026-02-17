import { useState } from "react";
import logo from "./assets/InSnapLogo.png";
import uploadIcon from "./assets/UploadImage.png";
import dropIcon from "./assets/DropIcon.png";
import clusterIcon from "./assets/ClusterIcon.png";
import getIcon from "./assets/GetPallete.png";


export default function PaletteGenerator() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [method, setMethod] = useState("kmeans");
  const [numColors, setNumColors] = useState(5);
  const [paletteMap, setPaletteMap] = useState({ kmeans: [], median_cut: [] });
  const [palette, setPalette] = useState([]);
  const [toast, setToast] = useState({ show: false, message: "" });

  // Handle image selection
  const handleImageChange = (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setPalette([]);
    setPaletteMap({ kmeans: [], median_cut: [] });
  };

  const handleMethodChange = (e) => {
    const newMethod = e.target.value;
    setMethod(newMethod);
    setPalette(paletteMap[newMethod] || []);
  };

  // Submit to backend
  const generatePalette = async () => {
    if (!image) {
      alert("Please upload or capture an image.");
      return;
    }

    const formData = new FormData();
    formData.append("image", image);
    formData.append("method", method);
    formData.append("num_colors", numColors);

    try {
      const response = await fetch("http://127.0.0.1:8000/generate", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setPalette(data.palette || []);
      setPaletteMap((prev) => ({
      ...prev,
      [method]: data.palette || [],
    }));
    } catch (error) {
      console.error("Error:", error);
    }
  };

  const copyHex = async (hex) => {
    try {
      await navigator.clipboard.writeText(hex);
      setToast({ show: true, message: `Copied ${hex} to clipboard!` });
      setTimeout(() => {
      setToast((prev) => ({ ...prev, show: false }));
    }, 1500);
    } catch (e) {
      console.error(e);
    }
  };

  const copyAll = async () => {
    if (!palette.length) return;
    copyHex(palette.join(", "));
  };

  const hasPicked = !!image;

  return (
    <div style={styles.page}>
      {/* Toast notification */}
      <div
        style={{
          ...styles.toast,
          ...(toast.show ? styles.toastVisible : styles.toastHidden),
        }}>
        {toast.message}
      </div>

      {/* NAVBAR */}
      <header style={styles.nav}>
        <div style={styles.navInner} className="insnap-navinner">
          <div style={styles.brand} className="insnap-brand">
            <img src={logo} alt="InSnap" style={styles.logo} />
          </div>

          <nav style={styles.links} className="insnap-links">
            <a className="navbar" href="#generate" style={styles.link}>
              Generate
            </a>
            <a className="navbar" href="#how" style={styles.link}>
              How it works
            </a>
            <a className="navbar" href="#about" style={styles.link}>
              About
            </a>
          </nav>

          <a
            href="https://github.com/DezJukes/colorpalettegen"
            target="_blank"
            rel="noopener noreferrer"
            style={styles.githubBtn}
            className="insnap-github"
          >
            View on GitHub
          </a>
        </div>
      </header>

      {/* MAIN */}
      <main style={styles.main} id="generate">
        <div style={styles.grid} className="insnap-grid">
          {/* LEFT CARD */}
          <section style={styles.card}>
            <h2 style={styles.cardTitle}>Upload an image</h2>

            <label style={styles.drop}>
              <input
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                style={styles.hiddenInput}
              />
              <div style={styles.dropBox}>
                <img src={dropIcon} alt="Drop" style={styles.dropIconImg} />
                <div style={styles.dropTextWrap}>
                  <div style={styles.dropText}>
                    Drop image here or{" "}
                    <span style={styles.dropBrowse}>Browse</span>
                  </div>
                  <div style={styles.dropSub}>JPG / PNG</div>
                </div>
              </div>
            </label>

            <div style={styles.btnRow}>
              <label style={{ ...styles.btn, ...styles.btnPrimary }}>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageChange}
                  style={styles.hiddenInput}
                />
                Browse File
              </label>

              <label style={{ ...styles.btn, ...styles.btnGhost }}>
                <input
                  type="file"
                  accept="image/*"
                  capture="environment"
                  onChange={handleImageChange}
                  style={styles.hiddenInput}
                />
                Use Camera
              </label>
            </div>

            <div style={styles.divider} />

            {/* Remove palette size buttons (as requested) */}

            {/* Keep slider (this is the palette size control) */}
            <div style={styles.fieldRowCol}>
              <div style={styles.fieldLabel}>Palette size: {numColors}</div>
              <input
                type="range"
                min="1"
                max="10"
                value={numColors}
                onChange={(e) => setNumColors(Number(e.target.value))}
                style={styles.range}
              />
            </div>

            <div style={styles.fieldRow} className="insnap-fieldrow">
              <div style={styles.fieldLabel}>Algorithm</div>
              <select
                value={method}
                onChange={handleMethodChange}
                style={styles.select}
              >
                <option value="kmeans">Fast K-Means Enhanced</option>
                <option value="median_cut">Fast K-Means Original</option>
              </select>
            </div>
          </section>

          {/* RIGHT CARD */}
          <section style={styles.card}>
            <div style={styles.cardHeader}>
              <h2 style={styles.cardTitle}>Generated Palette</h2>
              <button
                style={styles.moreBtn}
                onClick={copyAll}
                title="Copy all"
                type="button"
              >
                ⋯
              </button>
            </div>

            <div style={styles.previewWrap}>
              {preview ? (
                <img src={preview} alt="preview" style={styles.preview} />
              ) : (
                <div style={styles.previewEmpty}>
                  Upload an image to preview here.
                </div>
              )}
            </div>

            <div style={styles.paletteHeader}>
              <span style={styles.paletteHint}>Dominant → Accent</span>
              <span style={styles.paletteLine} />
            </div>

            <div style={styles.swatches}>
              {palette.length ? (
                palette.map((color, index) => (
                  <button className="card"
                    key={index}
                    style={{ ...styles.swatch, backgroundColor: color }}
                    onClick={() => copyHex(color)}
                    title="Tap to copy"
                    type="button"
                  >
                    <div style={styles.hex}>{color}</div>
                  </button>
                ))
              ) : (
                <div style={styles.emptyPalette}>
                  {hasPicked ? "Click Generate Palette." : "Pick an image first."}
                </div>
              )}
            </div>

            <button className="card" style={styles.bigGenerate} onClick={generatePalette}>
              Generate Palette
            </button>

            <div style={styles.actionsRow} className="insnap-actions">
              <button
                style={styles.actionBtn}
                onClick={() => alert("Add PNG export later")}
              >
                Download PNG
              </button>
              <button
                style={styles.actionBtn}
                onClick={() => alert("Add export options later")}
              >
                Export
              </button>
              <button style={styles.actionBtn} onClick={copyAll}>
                Copy All
              </button>
            </div>
          </section>
        </div>

        {/* Stats strip (only after pick) */}
        {hasPicked && (
          <div style={styles.stats}>
            <span style={styles.statItem}>
              Algorithm:{" "}
              <b>
                {method === "kmeans"
                  ? "Fast K-Means (Enhanced)"
                  : "Fast K-Means"}
              </b>
            </span>
            <span style={styles.sep}>|</span>
            <span style={styles.statItem}>
              k = <b>{numColors}</b>
            </span>
            <span style={styles.sep}>|</span>
            <span style={styles.statItem}>
              Colors extracted: <b>{palette.length}</b>
            </span>
          </div>
        )}

        {/* HOW IT WORKS */}
        <section style={styles.how} id="how">
          <h2 style={styles.howTitle}>How It Works</h2>

          <div style={styles.howGrid} className="insnap-howgrid">
            <div className="about" style={styles.howCard}>
              <img src={uploadIcon} alt="Upload" style={styles.howIconImg} />
              <div style={styles.howCardTitle}>Upload Image</div>
              <div style={styles.howCardText}>
                Import an image directly from your device.
              </div>
            </div>

            <div className="about" style={styles.howCard}>
              <img src={clusterIcon} alt="Cluster" style={styles.howIconImg} />
              <div style={styles.howCardTitle}>Cluster Colors</div>
              <div style={styles.howCardText}>
                Our Fast K-Means algorithm analyzes and clusters colors.
              </div>
            </div>

            <div className="about" style={styles.howCard}>
              <img src={getIcon} alt="Get Palette" style={styles.howIconImg2} />
              <div style={styles.howCardTitle}>Get Palette + Export</div>
              <div style={styles.howCardText}>
                Copy HEX values or export formats.
              </div>
            </div>
          </div>
        </section>

        {/* Footer (simple like mockup) */}
        <footer style={styles.footer} id="about">
          <span>InSnap © 2026</span>
          <span style={styles.footerSep}>|</span>
          <span>Research Project by D</span>
          <span style={styles.footerSep}>|</span>
          <span style={styles.footerLink} onClick={() => alert("Paper link here")}>
            Paper
          </span>
          <span style={styles.footerSep}>|</span>
          <span style={styles.footerLink} onClick={() => alert("Contact here")}>
            Contact
          </span>
        </footer>
      </main>

      {/* RESPONSIVE (phone) */}
      <style>{`
        @media (max-width: 980px) {
          .insnap-grid { grid-template-columns: 1fr !important; }
          .insnap-howgrid { grid-template-columns: 1fr !important; }
          .insnap-actions { grid-template-columns: 1fr !important; }
        }

        @media (max-width: 760px) {
          .insnap-links { display: none !important; }
          .insnap-github { display: none !important; }
          .insnap-navinner { height: 70px !important; }

          /* CENTER LOGO ON PHONE (GUARANTEED) */
          .insnap-navinner {
            justify-content: center !important;
            position: relative !important;
          }
          .insnap-brand {
            position: absolute !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: auto !important;
          }
        }

        @media (max-width: 520px) {
          .insnap-fieldrow { flex-direction: column !important; align-items: stretch !important; }
        }
      `}</style>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background: "#F6F7FB",
    fontFamily:
      'ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial',
    color: "#0F172A",
  },

  toast: {
  position: "fixed",
  top: "24px",
  left: "50%",
  transform: "translate(-50%, -20px)",
  background: "#2563EB",
  color: "#fff",
  padding: "12px 24px",
  borderRadius: "12px",
  fontWeight: 900,
  fontSize: "15px",
  zIndex: 9999,
  boxShadow: "0 4px 24px rgba(37,99,235,0.18)",
  opacity: 0,
  pointerEvents: "none",
  transition: "all 0.35s ease",
},

toastVisible: {
  opacity: 1,
  transform: "translate(-50%, 0px)",
  pointerEvents: "auto",
},

toastHidden: {
  opacity: 0,
  transform: "translate(-50%, -20px)",
},

  // NAV
  nav: {
    background: "#FFFFFF",
    borderBottom: "1px solid rgba(15,23,42,0.08)",
    position: "sticky",
    top: 0,
    zIndex: 10,
  },
  navInner: {
    maxWidth: "1100px",
    margin: "0 auto",
    height: "78px",
    padding: "0 18px",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "14px",
  },
  brand: { display: "flex", alignItems: "center" },

  // LOGO BIGGER (requested)
  logo: { height: "67px", width: "auto", display: "block" },

  links: { display: "flex", gap: "18px", alignItems: "center" },
  link: {
    textDecoration: "none",
    color: "rgba(15,23,42,0.65)",
    fontWeight: 700,
    fontSize: "14px",
  },
  githubBtn: {
    textDecoration: "none",
    padding: "10px 14px",
    borderRadius: "14px",
    background: "#F1F5F9",
    border: "1px solid rgba(15,23,42,0.08)",
    color: "#0F172A",
    fontWeight: 800,
    fontSize: "14px",
  },

  // MAIN
  main: { maxWidth: "1100px", margin: "0 auto", padding: "28px 18px 36px" },

  grid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "18px",
    alignItems: "start",
  },

  card: {
    background: "#FFFFFF",
    borderRadius: "18px",
    padding: "18px",
    border: "1px solid rgba(15,23,42,0.08)",
    boxShadow: "0 14px 30px rgba(15,23,42,0.08)",
    display: "flex",
    flexDirection: "column",
    gap: "12px",
  },

  cardHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: "10px",
  },
  moreBtn: {
    width: "40px",
    height: "40px",
    borderRadius: "12px",
    border: "1px solid rgba(15,23,42,0.10)",
    background: "#fff",
    cursor: "pointer",
    fontSize: "18px",
  },

  cardTitle: { margin: 0, fontSize: "22px", fontWeight: 900 },

  // Upload box
  hiddenInput: { display: "none" },
  drop: { cursor: "pointer" },
  dropBox: {
    border: "2px dashed rgba(59,130,246,0.22)",
    borderRadius: "16px",
    padding: "18px",
    background: "linear-gradient(180deg, #FBFCFF 0%, #FFFFFF 100%)",
    display: "flex",
    alignItems: "center",
    gap: "12px",
  },
  dropIconImg: {
    width: "48px",
    height: "48px",
    borderRadius: "14px",
    background: "rgba(59,130,246,0.10)",
    border: "1px solid rgba(59,130,246,0.16)",
  },
  dropTextWrap: { display: "flex", flexDirection: "column", gap: "4px" },
  dropText: { fontWeight: 900, fontSize: "14px" },
  dropBrowse: { color: "#2563EB" },
  dropSub: { fontSize: "12px", color: "rgba(15,23,42,0.55)" },

  btnRow: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" },
  btn: {
    padding: "12px 12px",
    borderRadius: "14px",
    border: "1px solid rgba(15,23,42,0.10)",
    background: "#fff",
    cursor: "pointer",
    fontWeight: 900,
    textAlign: "center",
    userSelect: "none",
  },
  btnPrimary: {
    background: "linear-gradient(90deg, #2563EB 0%, #3B82F6 100%)",
    color: "#fff",
    border: "1px solid rgba(37,99,235,0.22)",
  },
  btnGhost: { background: "#F8FAFC" },

  divider: { height: "1px", background: "rgba(15,23,42,0.08)", margin: "6px 0" },

  fieldRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "12px",
  },
  fieldRowCol: { display: "flex", flexDirection: "column", gap: "10px" },
  fieldLabel: { fontWeight: 800, color: "rgba(15,23,42,0.75)", fontSize: "14px" },

  select: {
    padding: "10px 12px",
    borderRadius: "12px",
    border: "1px solid rgba(15,23,42,0.10)",
    background: "#fff",
    fontWeight: 800,
    outline: "none",
    minWidth: "220px",
  },
  range: { width: "100%" },

  // Preview + palette
  previewWrap: {
    borderRadius: "16px",
    overflow: "hidden",
    border: "1px solid rgba(15,23,42,0.08)",
    background: "#F8FAFC",
  },
  preview: { width: "100%", height: "240px", objectFit: "cover", display: "block" },
  previewEmpty: {
    height: "240px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "rgba(15,23,42,0.55)",
    fontWeight: 700,
    textAlign: "center",
    padding: "10px",
  },

  paletteHeader: { display: "flex", alignItems: "center", gap: "10px", marginTop: "4px" },
  
  paletteHint: { 
    fontSize: "12px", 
    fontWeight: 800, 
    letterSpacing: "0.5px",
    textTransform: "uppercase",
    color: "rgba(15,23,42,0.55)" 
  },

  paletteLine: { height: "1px", flex: 1, background: "rgba(15,23,42,0.08)" },

  swatches: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(90px, 1fr))",
    gap: "14px",
    marginTop: "12px",
    minHeight: "90px",
  },
  swatch: {
    width: "100%",
    aspectRatio: "1 / 1",
    borderRadius: "16px",
    border: "1px solid rgba(15,23,42,0.08)",
    cursor: "pointer",
    display: "flex",
    alignItems: "flex-end",
    justifyContent: "center",
    padding: "10px",
    backgroundClip: "padding-box",
    transition: "transform 0.15s ease, box-shadow 0.15s ease",
  },

  hex: {
    background: "rgba(255,255,255,0.9)",
    padding: "6px 10px",
    borderRadius: "999px",
    fontSize: "11px",
    fontWeight: 900,
    color: "#0F172A",
    letterSpacing: "0.5px",
  },

  emptyPalette: { color: "rgba(15,23,42,0.55)", fontWeight: 700 },

  bigGenerate: {
    marginTop: "6px",
    padding: "14px 14px",
    borderRadius: "16px",
    border: "1px solid rgba(37,99,235,0.22)",
    background: "linear-gradient(90deg, #2563EB 0%, #3B82F6 100%)",
    color: "#fff",
    cursor: "pointer",
    fontWeight: 1000,
    fontSize: "15px",
  },

  actionsRow: { display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "10px" },
  actionBtn: {
    padding: "11px 12px",
    borderRadius: "14px",
    border: "1px solid rgba(15,23,42,0.10)",
    background: "#F8FAFC",
    cursor: "pointer",
    fontWeight: 900,
  },

  // Stats strip
  stats: {
    marginTop: "18px",
    background: "#FFFFFF",
    border: "1px solid rgba(15,23,42,0.08)",
    borderRadius: "16px",
    padding: "14px 14px",
    boxShadow: "0 14px 30px rgba(15,23,42,0.06)",
    display: "flex",
    flexWrap: "wrap",
    gap: "10px",
    justifyContent: "center",
    color: "rgba(15,23,42,0.70)",
  },
  statItem: { fontWeight: 700 },
  sep: { opacity: 0.35 },

  // How it works
  how: { marginTop: "42px" },
  howTitle: { textAlign: "center", fontSize: "34px", fontWeight: 1000, margin: "0 0 18px" },
  howGrid: { display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "18px" },
  howCard: {
    background: "#FFFFFF",
    border: "1px solid rgba(15,23,42,0.08)",
    borderRadius: "18px",
    padding: "18px",
    boxShadow: "0 14px 30px rgba(15,23,42,0.06)",
    textAlign: "center",
  },
  howIcon: {
    width: "52px",
    height: "52px",
    borderRadius: "16px",
    background: "rgba(59,130,246,0.10)",
    border: "1px solid rgba(59,130,246,0.16)",
    margin: "0 auto 12px",
  },
  howCardTitle: { fontWeight: 1000, fontSize: "16px", marginBottom: "6px" },
  howCardText: { color: "rgba(15,23,42,0.60)", fontWeight: 600, fontSize: "13px" },

  // Footer
  footer: {
    marginTop: "46px",
    paddingTop: "18px",
    color: "rgba(15,23,42,0.55)",
    display: "flex",
    justifyContent: "center",
    gap: "10px",
    flexWrap: "wrap",
    fontWeight: 700,
    fontSize: "13px",
  },
  footerSep: { opacity: 0.35 },
  footerLink: { cursor: "pointer", textDecoration: "underline" },

  howIconImg: {
    width: "72px",
    height: "72px",
    objectFit: "contain",
    display: "block",
    margin: "0 auto 12px",
  },

  howIconImg2: {
    width: "100px",
    height: "100px",
    objectFit: "contain",
    display: "block",
    margin: "0 auto -12px",
  },
  
};

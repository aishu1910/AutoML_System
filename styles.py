def load_css():
    return """
    <style>
    section[data-testid="stFileUploader"] {
    border: 5px solid red !important;
     }
     .stDownloadButton > button {
    background-color: #cba6f7 !important;
    color: black !important;   /* 👈 your requirement */
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
    font-weight: 600;
}

/* Hover */
   .stDownloadButton > button:hover {
    background-color: #b48ef2 !important;
    color: black !important;
}
     /* Upload container */
    [data-testid="stFileUploader"] {
    background-color: #313244;
    border: 2px dashed #cba6f7;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
}
   
/* Inner dropzone */
    [data-testid="stFileDropzone"] {
    background-color: #ffffff !important;
    border-radius: 10px;
    padding: 20px;
}

/* Improve text visibility */
   [data-testid="stFileUploader"] span,
   [data-testid="stFileUploader"] p,
   [data-testid="stFileUploader"] small {
   color: #000000 !important;
   font-weight: 500;
}
    /* ===== GLOBAL THEME ===== */
    .stApp {
        background-color: #1e1e2e;
        color: #ffffff;
    }

    .stSidebar {
        background-color: #181825;
    }

    h1, h2, h3, h4, h5, h6, p, label {
        color: #ffffff;
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        background-color: #cba6f7;
        color: #1e1e2e;
        border-radius: 8px;
        border: none;
    }

    /* ===== FILE UPLOADER (CLEAN FIX) ===== */

    /* Make outer dark */
    [data-testid="stFileUploader"] {
        background-color: #313244;
        border: 2px dashed #cba6f7;
        border-radius: 10px;
        padding: 20px;
    }

    /* Force inner box white */
    [data-testid="stFileDropzone"] {
        background-color: #ffffff !important;
        border-radius: 10px;
    }

    /* 🔥 FORCE BLACK TEXT (strong override) */
    section[data-testid="stFileUploader"] div,
    section[data-testid="stFileUploader"] span,
    section[data-testid="stFileUploader"] p,
    section[data-testid="stFileUploader"] small {
        color: black !important;
        -webkit-text-fill-color: black !important;
    }

    </style>
    """
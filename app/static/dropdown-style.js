// Function to style dropdowns
function styleDropdowns() {
    // Create and append a style element
    const style = document.createElement('style');
    style.textContent = `
        /* Force light green background on all dropdown elements */
        .stSelectbox > div > div > div,
        .stSelectbox > div > div > div > div,
        .stSelectbox > div > div > div > div > div,
        .stSelectbox > div > div > div > div > div > div,
        .stSelectbox > div > div > div > div > div > div > div,
        .stSelectbox > div > div > div > div > div > div > div > div,
        .stSelectbox [class*='-control'],
        .stSelectbox [class*='-menu'],
        .stSelectbox [class*='-option'],
        .stSelectbox [class*='-singleValue'],
        .stSelectbox [class*='-placeholder'],
        .stSelectbox [class*='-input'] {
            background-color: #f0fdf4 !important;
            color: #111827 !important;
            border-color: #bbf7d0 !important;
        }
        
        /* Hover state */
        .stSelectbox [class*='-option']:hover {
            background-color: #dcfce7 !important;
        }
    `;
    document.head.appendChild(style);
    
    // Also directly set styles on all elements
    document.querySelectorAll('.stSelectbox *').forEach(el => {
        el.style.setProperty('background-color', '#f0fdf4', 'important');
        el.style.setProperty('color', '#111827', 'important');
    });
}

// Run on page load
document.addEventListener('DOMContentLoaded', styleDropdowns);

// Also run after Streamlit updates the DOM
const observer = new MutationObserver(styleDropdowns);
observer.observe(document.body, { childList: true, subtree: true });

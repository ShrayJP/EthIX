
console.log("Dark Pattern Detector: Content Script Loaded");

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "analyze") {
        analyzePage();
    }
});

async function analyzePage() {
    console.log("Starting analysis...");
    
    // 1. Scrape visible text elements
    // We target common text containers
    const elementsToAnalyze = [];
    const domElements = document.querySelectorAll("p, h1, h2, h3, h4, h5, span, div, li, a, button, label");
    
    // Assign unique IDs to elements for tracking
    let idCounter = 0;
    
    domElements.forEach(el => {
        // Skip hidden elements or tiny text
        if (el.offsetParent === null) return;
        const text = el.innerText.trim();
        if (text.length < 5) return; // Ignore very short text
        if (text.length > 500) return; // Ignore huge blocks for now
        
        // Check if element already has ID or assign one
        if (!el.dataset.dpId) {
            el.dataset.dpId = `dp-${idCounter++}`;
        }
        
        elementsToAnalyze.push({
            id: el.dataset.dpId,
            text: text,
            type: el.tagName.toLowerCase()
        });
    });
    
    console.log(`Sending ${elementsToAnalyze.length} elements to server...`);
    
    // 2. Send to local Python server
    try {
        const response = await fetch("http://localhost:5000/analyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                url: window.location.href,
                elements: elementsToAnalyze
            })
        });
        
        const data = await response.json();
        console.log("Analysis complete:", data);
        
        // 3. Highlight results
        highlightResults(data.results);
        
        // Send summary back to popup
        chrome.runtime.sendMessage({
            action: "analysisComplete",
            count: data.count,
            results: data.results
        });
        
    } catch (error) {
        console.error("Analysis failed:", error);
        chrome.runtime.sendMessage({
            action: "analysisError",
            error: error.message
        });
    }
}

function highlightResults(results) {
    if (!results) return;
    
    results.forEach(item => {
        const el = document.querySelector(`[data-dp-id="${item.id}"]`);
        if (el) {
            // Apply red border
            el.style.border = "3px solid red";
            el.style.position = "relative";
            
            // Create a tooltip/label
            const label = document.createElement("div");
            label.innerText = `⚠️ ${item.patterns.join(", ")}`;
            label.style.position = "absolute";
            label.style.top = "-25px";
            label.style.left = "0";
            label.style.backgroundColor = "red";
            label.style.color = "white";
            label.style.padding = "2px 5px";
            label.style.fontSize = "12px";
            label.style.borderRadius = "3px";
            label.style.zIndex = "10000";
            label.style.pointerEvents = "none"; // Make sure label doesn't block clicks
            
            // Append label near the element
            // Note: Appending directly inside might break layout, but good for demo
            // Better to use a separate overlay container, but this is simpler
            el.parentNode.insertBefore(label, el);
        }
    });
}

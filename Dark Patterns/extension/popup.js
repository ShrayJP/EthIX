
document.getElementById('analyzeBtn').addEventListener('click', async () => {
    // 1. Get current active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    // 2. Show loading state
    const statusDiv = document.getElementById('status');
    const resultsDiv = document.getElementById('results');
    statusDiv.innerText = "Analyzing page content...";
    statusDiv.style.color = "#007bff";
    resultsDiv.innerText = "";

    // 3. Inject content script (if not already injected)
    // For MV3, we usually rely on content_scripts declared in manifest, 
    // but we can also execute script programmatically to be safe or to trigger
    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ['content.js']
    }, () => {
        // Send message to content script to start analysis
        chrome.tabs.sendMessage(tab.id, { action: "analyze" });
    });
});

// Listen for results from content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    const statusDiv = document.getElementById('status');
    const resultsDiv = document.getElementById('results');

    if (message.action === "analysisComplete") {
        statusDiv.innerText = "Analysis Complete!";
        statusDiv.style.color = "green";

        if (message.count > 0) {
            resultsDiv.innerHTML = `Found <b style='color:red;'>${message.count}</b> potential dark patterns.<br>Check the page for red highlights.`;
        } else {
            resultsDiv.innerText = "No dark patterns detected based on current text.";
        }
    } else if (message.action === "analysisError") {
        statusDiv.innerText = "Error during analysis.";
        statusDiv.style.color = "red";
        resultsDiv.innerText = message.error;
    }
});

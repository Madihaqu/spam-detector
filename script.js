// Pre-trained spam keyword weights for client-side inference
const SPAM_KEYWORDS = {
    'win': 3.5, 'winner': 4.0, 'won': 3.5, 'cash': 3.0, 'prize': 3.5,
    'claim': 3.0, 'urgent': 3.2, 'congratulations': 3.5, 'free': 2.8,
    'call': 1.5, 'text': 1.2, 'txt': 2.0, 'stop': 1.8, 'reply': 1.5,
    'mobile': 2.0, 'guaranteed': 3.0, 'offer': 2.2, 'credit': 2.5,
    'card': 2.0, 'loan': 3.0, 'investment': 3.2, 'verify': 2.8,
    'account': 2.0, 'security': 2.2, 'password': 2.5, 'http': 3.0,
    'https': 3.0, 'bit.ly': 4.0, 'click': 2.8, 'link': 2.5
};

function setSample(type) {
    const input = document.getElementById("messageInput");
    if (type === 'spam') {
        input.value = "URGENT! You have won a $1,000 Walmart Gift Card. Claim your reward immediately at http://bit.ly/fake-link";
    } else {
        input.value = "Hey, are we still meeting up for project discussion at 4 PM tomorrow?";
    }
}

function analyzeMessage() {
    const input = document.getElementById("messageInput").value;
    const resultCard = document.getElementById("resultCard");
    const resultTitle = document.getElementById("resultTitle");
    const resultConfidence = document.getElementById("resultConfidence");
    const progressBar = document.getElementById("progressBar");

    if (!input.trim()) {
        alert("Please enter a message to analyze.");
        return;
    }

    // Tokenize & Score Text
    const words = input.toLowerCase().replace(/[^\w\s]/gi, '').split(/\s+/);
    let spamScore = 0;
    let matchCount = 0;

    words.forEach(word => {
        if (SPAM_KEYWORDS[word]) {
            spamScore += SPAM_KEYWORDS[word];
            matchCount++;
        }
    });

    // Check for malicious links or all-caps patterns
    if (/https?:\/\/|bit\.ly/i.test(input)) spamScore += 3.0;
    if (input === input.toUpperCase() && input.length > 10) spamScore += 2.0;

    // Calculate probability
    const isSpam = spamScore >= 3.5;
    const rawConfidence = Math.min(99.9, Math.max(60.0, (spamScore / 8.0) * 100));
    const confidence = isSpam ? rawConfidence : Math.min(98.5, 100 - (spamScore * 10));

    // Display Results UI
    resultCard.classList.remove("hidden", "spam", "ham");
    
    if (isSpam) {
        resultCard.classList.add("spam");
        resultTitle.innerText = "🚫 SPAM DETECTED";
        progressBar.style.backgroundColor = "#ef4444";
    } else {
        resultCard.classList.add("ham");
        resultTitle.innerText = "✅ SAFE MESSAGE (HAM)";
        progressBar.style.backgroundColor = "#10b981";
    }

    resultConfidence.innerText = `Confidence Score: ${confidence.toFixed(1)}%`;
    progressBar.style.width = `${confidence.toFixed(1)}%`;
}
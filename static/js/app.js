// =========================================================
// CHURNSHIELD AI — CUSTOMER CHURN PREDICTION
// =========================================================

const form = document.getElementById("predictionForm");

const resultPlaceholder =
    document.getElementById("resultPlaceholder");

const resultContent =
    document.getElementById("resultContent");

    // Update probability ring
const resultCard =
    document.querySelector(".result-card");

resultCard.style.setProperty(
    "--probability",
    probability
);

const probabilityElement =
    document.getElementById("probability");

const riskLevelElement =
    document.getElementById("riskLevel");

const predictionTextElement =
    document.getElementById("predictionText");

const predictionDescriptionElement =
    document.getElementById("predictionDescription");


// =========================================================
// FORM SUBMISSION
// =========================================================

form.addEventListener("submit", async function (event) {

    event.preventDefault();

    // -----------------------------------------
    // Collect latest customer values
    // -----------------------------------------

    const customerData = {
        credit_score: Number(
            document.getElementById("credit_score").value
        ),

        age: Number(
            document.getElementById("age").value
        ),

        tenure: Number(
            document.getElementById("tenure").value
        ),

        balance: Number(
            document.getElementById("balance").value
        ),

        products_number: Number(
            document.getElementById("products_number").value
        ),

        country:
            document.getElementById("country").value,

        gender:
            document.getElementById("gender").value,

        estimated_salary: Number(
            document.getElementById("estimated_salary").value
        ),

        credit_card:
            document.getElementById("credit_card").checked ? 1 : 0,

        active_member:
            document.getElementById("active_member").checked ? 1 : 0
    };


    // -----------------------------------------
    // Validate numeric values
    // -----------------------------------------

    if (
        !Number.isFinite(customerData.credit_score) ||
        !Number.isFinite(customerData.age) ||
        !Number.isFinite(customerData.tenure) ||
        !Number.isFinite(customerData.balance) ||
        !Number.isFinite(customerData.products_number) ||
        !Number.isFinite(customerData.estimated_salary)
    ) {
        alert("Please enter valid customer information.");
        return;
    }

    const validationError = validateCustomerData(customerData);

    if (validationError) {
       alert(validationError);
       return;
}


    // -----------------------------------------
    // Loading state
    // -----------------------------------------

    const button =
        form.querySelector(".predict-btn");

    const buttonText =
        button.querySelector("span");

    button.disabled = true;

if (buttonText) {
    buttonText.innerHTML = `
        <span class="loader"></span>
        Analyzing...
    `;
}


    try {

        // -----------------------------------------
        // Send latest data to Flask
        // -----------------------------------------

        const response = await fetch("/predict", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(customerData)
        });


        // -----------------------------------------
        // Check server response
        // -----------------------------------------

        if (!response.ok) {
            throw new Error(
                `Server error: ${response.status}`
            );
        }


        const result = await response.json();


        if (!result.success) {
            throw new Error(
                result.error || "Prediction failed."
            );
        }


        // -----------------------------------------
        // Get NEW prediction
        // -----------------------------------------

        const probability =
            Number(result.probability);

        const prediction =
            Number(result.prediction);

        const risk =
            String(result.risk || "Low");


        if (!Number.isFinite(probability)) {
            throw new Error(
                "Invalid probability received from server."
            );
        }


        // -----------------------------------------
        // Update probability
        // -----------------------------------------

        probabilityElement.textContent =
            probability.toFixed(2);


        // -----------------------------------------
        // Update risk level
        // -----------------------------------------

        riskLevelElement.textContent =
            risk.toUpperCase();


        // -----------------------------------------
        // Update prediction message
        // -----------------------------------------

        if (prediction === 1) {

            predictionTextElement.textContent =
                "Customer At Risk";

            predictionDescriptionElement.textContent =
                `The model estimates a ${probability.toFixed(2)}% probability that this customer may churn.`;

        } else {

            predictionTextElement.textContent =
                "Customer Likely to Stay";

            predictionDescriptionElement.textContent =
                `The model estimates a ${probability.toFixed(2)}% probability that this customer may churn.`;
        }


        // -----------------------------------------
// Dynamic Risk Styling
// -----------------------------------------

const resultCard =
    document.querySelector(".result-card");

// Remove previous risk class
resultCard.classList.remove(
    "risk-low",
    "risk-medium",
    "risk-high",
    "risk-very-high"
);

// Normalize risk returned by Flask
const normalizedRisk =
    risk.toLowerCase().trim();


// Apply new risk class
if (normalizedRisk === "very high") {

    resultCard.classList.add("risk-very-high");

} else if (normalizedRisk === "high") {

    resultCard.classList.add("risk-high");

} else if (normalizedRisk === "medium") {

    resultCard.classList.add("risk-medium");

} else {

    resultCard.classList.add("risk-low");
}


        // -----------------------------------------
        // Show latest result
        // -----------------------------------------

        resultPlaceholder.classList.add("hidden");

        resultContent.classList.remove("hidden");


        // -----------------------------------------
        // Add latest prediction to history
        // -----------------------------------------

        addPredictionHistory(
            probability,
            risk,
            prediction
        );


        // -----------------------------------------
        // Debug information
        // -----------------------------------------

        console.log(
            "Customer submitted:",
            customerData
        );

        console.log(
            "Latest prediction:",
            result
        );


    } catch (error) {

        console.error(
            "Prediction error:",
            error
        );

        alert(
            "Prediction failed. Please check the server and try again."
        );

    } finally {

        button.disabled = false;

        if (buttonText) {
           buttonText.textContent = "Analyze Customer";
    } 
    }
});


// =========================================================
// RESET / ANALYZE ANOTHER CUSTOMER
// =========================================================

function resetPrediction() {
    const resultContent = document.getElementById("resultContent");
    const resultPlaceholder = document.getElementById("resultPlaceholder");
    const predictionForm = document.getElementById("predictionForm");
    const resultCard = document.querySelector(".result-card");

    // Hide result and show placeholder
    resultContent.classList.add("hidden");
    resultPlaceholder.classList.remove("hidden");

    // Reset form
    predictionForm.reset();

    // Restore default toggle values
    document.getElementById("credit_card").checked = true;
    document.getElementById("active_member").checked = true;

    // Reset result values
    probabilityElement.textContent = "0";
    riskLevelElement.textContent = "LOW";

    predictionTextElement.textContent = "Low Churn Risk";

    predictionDescriptionElement.textContent =
        "Enter customer information and analyze the customer again.";

    // Reset probability ring
    resultCard.style.setProperty("--probability", 0);

    // Reset risk classes
    resultCard.classList.remove(
        "risk-low",
        "risk-medium",
        "risk-high",
        "risk-very-high"
    );

    // Restore default risk style
    resultCard.classList.add("risk-low");

    // Scroll back to input section
    document.querySelector(".input-card").scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


// =========================================================
// PREDICTION HISTORY
// =========================================================

function addPredictionHistory(
    probability,
    risk,
    prediction
) {

    const historyList =
        document.getElementById("historyList");


    const emptyHistory =
        historyList.querySelector(".empty-history");


    if (emptyHistory) {
        emptyHistory.remove();
    }


    const item =
        document.createElement("div");

    item.className =
        "history-item";


    const status =
        prediction === 1
            ? "At Risk"
            : "Likely to Stay";


    item.innerHTML = `

        <div class="history-info">

            <strong>
                ${status}
            </strong>

            <span>
                Churn probability:
                ${Number(probability).toFixed(2)}%
            </span>

        </div>

        <div class="history-risk">
            ${risk}
        </div>

    `;


    historyList.prepend(item);

    const predictionCount =
    document.getElementById("predictionCount");

    if (predictionCount) {
       const currentCount =
        historyList.querySelectorAll(".history-item").length;

    predictionCount.textContent = currentCount;
}
}


// =========================================================
// CLEAR HISTORY
// =========================================================

function clearHistory() {

    const historyList =
        document.getElementById("historyList");


    historyList.innerHTML = `

        <div class="empty-history">
            No predictions yet.
        </div>

    `;

    const predictionCount =
    document.getElementById("predictionCount");

    if (predictionCount) {
       predictionCount.textContent = "0";
}
}

function validateCustomerData(data) {
    if (data.credit_score < 300 || data.credit_score > 850) {
        return "Credit score must be between 300 and 850.";
    }

    if (data.age < 18 || data.age > 100) {
        return "Age must be between 18 and 100.";
    }

    if (data.tenure < 0 || data.tenure > 10) {
        return "Tenure must be between 0 and 10 years.";
    }

    if (data.balance < 0) {
        return "Balance cannot be negative.";
    }

    if (data.products_number < 1 || data.products_number > 4) {
        return "Products number must be between 1 and 4.";
    }

    if (data.estimated_salary < 0) {
        return "Estimated salary cannot be negative.";
    }

    return null;
}

function addPredictionHistory(probability, risk, prediction) {
    const historyList = document.getElementById("historyList");

    if (!historyList) {
        return;
    }

    const emptyHistory = historyList.querySelector(".empty-history");

    if (emptyHistory) {
        emptyHistory.remove();
    }

    const item = document.createElement("div");
    const riskClass = risk.toLowerCase().replace(/\s+/g, "-");

    item.className = `history-item history-${riskClass}`;

    const status = prediction === 1
        ? "At Risk"
        : "Likely to Stay";

    const now = new Date();

    const time = now.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit"
    });

    item.innerHTML = `
        <div class="history-info">
            <strong>${status}</strong>

            <span>
                Churn probability:
                ${Number(probability).toFixed(2)}%
            </span>

            <small>
                Analyzed at ${time}
            </small>
        </div>

        <div class="history-risk">
            ${risk}
        </div>
    `;

    historyList.prepend(item);
}
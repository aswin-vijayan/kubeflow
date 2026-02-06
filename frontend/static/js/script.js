document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    const data = {};

    formData.forEach((val, key) => data[key] = Number(val));

    const yearsAtCompany = data['Years at Company'];
    const companyTenure = data['Company Tenure'];

    data["RoleStagnationRatio"] = Number((yearsAtCompany / (companyTenure + 1)).toFixed(3));
    data["TenureGap"] = Number((companyTenure - yearsAtCompany).toFixed(2));
    data["EarlyCompanyTenureRisk"] = yearsAtCompany <= 2 ? 1 : 0;
    data["LongTenureLowRoleRisk"] = (companyTenure > 5 && data["Job Level"] <= 2) ? 1 : 0;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });

        const result = await response.json();
        console.log(result)

        const showResult = document.getElementById("result")
        showResult.style.display = 'block'
        showResult.innerHTML = `
            <h2 class="text-primary mb-3 text-center">Prediction Result</h2>

            <div class="text-center mb-3">
                <h4 class="fw-bold ${result.prediction === 1 ? "text-danger" : "text-success"}">
                    ${result.prediction === 1 ? "Leave" : "Stay"}
                </h4>
                <hr/>
                <h6 class="text-muted">Probability</h6>
                <h4 class="text-danger fw-bold">${(result.probs * 100).toFixed(2)}%</h4>
            </div>
        `;
    } catch (error) {
        console.error("Prediction error:", error);
        document.getElementById("result").innerHTML =
            `<p class="text-danger">Prediction failed. Please try again.</p>`;
    }
});

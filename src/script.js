function predict() {
    // Show loading message
    $('#loading').show();
    console.log("predicting");

    // Get user inputs
    const age = parseFloat($('#age').val());
    const sex = parseFloat($('input[name="sex"]:checked').val());
    const cp = parseFloat($('#cp').val());
    const trestbps = parseFloat($('#trestbps').val());
    const chol = parseFloat($('#chol').val());
    const fbs = parseFloat($('#fbs').val());
    const restecg = parseFloat($('#restecg').val());
    const thalach = parseFloat($('#thalach').val());
    
    // Update the way exang is retrieved
    const exang = parseFloat($('input[name="exang"]:checked').val());

    const oldpeak = parseFloat($('#oldpeak').val());
    const slope = parseFloat($('#slope').val());
    const ca = parseFloat($('#ca').val());
    const thal = parseFloat($('#thal').val());
    console.log("past inputs");

    // Prepare input data for prediction
    const inputData = { age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal };
    console.log("about to make predictions");

    // Make predictions by sending data to the Node.js server
    $.ajax({
        type: 'POST',
        url: 'http://localhost:3000/predict',
        contentType: 'application/json',
        data: JSON.stringify(inputData),
        success: function (response) {
            console.log("Success:", response);

            const predictionsElement = $('#predictions');
            const loadingElement = $('#loading');

            // Hide loading message
            loadingElement.hide();

            // Display predictions
            predictionsElement.html(`
                <h2>Predictions:</h2>
                <p>k-NN Prediction: ${response.predictions.knn_prediction}</p>
                <p>Linear Regression Prediction: ${response.predictions.lr_prediction}</p>
                <p>Random Forest Prediction: ${response.predictions.rf_prediction}</p>
                <p>Stacked Ensemble Prediction: ${response.predictions.stacked_ensemble_prediction}</p>
                <p>Weighted Ensemble Prediction: ${response.predictions.weighted_ensemble_prediction}</p>
                <p>XGBoost Prediction: ${response.predictions.xgb_prediction}</p>
            `);
        },
        error: function (error) {
            console.error('Error:', error);

            // Hide loading message on error
            $('#loading').hide();
        },
        complete: function () {
            console.log("AJAX request complete");
        }
    });

    console.log("done!");
}


function showWeather() {
    var city = document.getElementById("city").value;

    // Simulate fetching weather data (replace with actual API call)
    var temperatureData = "25°C";
    var humidityData = "60%";
    var windData = "10 km/h";
    var rainfallData = "Chưa biết đơn vị"

    // Display weather data
    document.getElementById("temperature").textContent = temperatureData;
    document.getElementById("humidity").textContent = humidityData;
    document.getElementById("wind").textContent = windData;
    document.getElementById("rainfall").textContent = rainfallData;

    // Show the weather data container
    document.getElementById("weatherData").style.display = "block";
}

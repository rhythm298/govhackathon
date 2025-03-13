import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle } from 'react-leaflet';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar } from 'recharts';
import axios from 'axios';
import { 
  Card, CardContent, Typography, Grid, Button, FormControl, 
  InputLabel, Select, MenuItem, Box, Paper, Divider, 
  CircularProgress, Alert
} from '@mui/material';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix for Leaflet marker icons in React
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Sample traffic data
const generateTrafficData = () => {
  const areas = [
    { id: 1, name: "Downtown", lat: 40.7128, lng: -74.006, congestionLevel: 75 },
    { id: 2, name: "Midtown", lat: 40.7549, lng: -73.9840, congestionLevel: 60 },
    { id: 3, name: "Upper East Side", lat: 40.7735, lng: -73.9565, congestionLevel: 30 },
    { id: 4, name: "Brooklyn Heights", lat: 40.6959, lng: -73.9932, congestionLevel: 45 },
    { id: 5, name: "Queens Center", lat: 40.7337, lng: -73.8694, congestionLevel: 55 }
  ];

  return areas;
};

// Sample historical data
const generateHistoricalData = () => {
  const hours = [];
  for (let i = 0; i < 24; i++) {
    const morningPeak = Math.sin((i - 8) * Math.PI / 8) * 40 + 40;
    const eveningPeak = Math.sin((i - 18) * Math.PI / 8) * 35 + 35;
    
    const value = i >= 7 && i <= 10 ? morningPeak : 
                 i >= 16 && i <= 19 ? eveningPeak : 
                 Math.random() * 20 + 20;
    
    hours.push({
      hour: `${i}:00`,
      congestion: Math.floor(value),
      incidents: Math.floor(Math.random() * 5)
    });
  }
  return hours;
};

// Sample prediction data
const generatePredictions = () => {
  const hours = [];
  const currentHour = new Date().getHours();
  
  for (let i = 0; i < 8; i++) {
    const hour = (currentHour + i) % 24;
    const morningPeak = Math.sin((hour - 8) * Math.PI / 8) * 40 + 40;
    const eveningPeak = Math.sin((hour - 18) * Math.PI / 8) * 35 + 35;
    
    const value = hour >= 7 && hour <= 10 ? morningPeak : 
                 hour >= 16 && hour <= 19 ? eveningPeak : 
                 Math.random() * 20 + 20;
    
    hours.push({
      hour: `${hour}:00`,
      predicted: Math.floor(value)
    });
  }
  return hours;
};

// Sample incidents data
const generateIncidents = () => {
  const incidents = [
    { id: 1, type: "Accident", location: "Main St & 5th Ave", status: "Clearing", lat: 40.7282, lng: -73.9942, severity: "High" },
    { id: 2, type: "Construction", location: "Broadway & 42nd St", status: "Ongoing", lat: 40.7557, lng: -73.9862, severity: "Medium" },
    { id: 3, type: "Roadblock", location: "Park Ave & 23rd St", status: "Resolved", lat: 40.7405, lng: -73.9878, severity: "Low" }
  ];
  return incidents;
};

export default function TrafficDashboard() {
  const [trafficData, setTrafficData] = useState([]);
  const [historicalData, setHistoricalData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [incidents, setIncidents] = useState([]);
  const [selectedArea, setSelectedArea] = useState('all');
  const [timeRange, setTimeRange] = useState('day');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [mapCenter, setMapCenter] = useState([40.7128, -74.006]);
  const [mapZoom, setMapZoom] = useState(12);

  useEffect(() => {
    // Simulate API fetching with a delay
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // In a real app, you would fetch data from APIs
        // For example:
        // const trafficResponse = await axios.get('/api/traffic');
        // const historicalResponse = await axios.get('/api/historical');
        
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Set sample data
        setTrafficData(generateTrafficData());
        setHistoricalData(generateHistoricalData());
        setPredictions(generatePredictions());
        setIncidents(generateIncidents());
        
        setLoading(false);
      } catch (err) {
        setError("Failed to load traffic data. Please try again later.");
        setLoading(false);
        console.error("Error fetching traffic data:", err);
      }
    };

    fetchData();
  }, []);

  // Function to handle area selection
  const handleAreaChange = (event) => {
    setSelectedArea(event.target.value);
    
    if (event.target.value !== 'all') {
      const area = trafficData.find(a => a.id === event.target.value);
      if (area) {
        setMapCenter([area.lat, area.lng]);
        setMapZoom(14);
      }
    } else {
      setMapCenter([40.7128, -74.006]);
      setMapZoom(12);
    }
  };

  // Get congestion color based on level
  const getCongestionColor = (level) => {
    if (level < 30) return '#4CAF50'; // Green
    if (level < 60) return '#FFC107'; // Yellow
    return '#F44336'; // Red
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
        <Typography variant="h6" sx={{ ml: 2 }}>Loading traffic data...</Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Traffic Congestion Management Dashboard
      </Typography>
      
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <FormControl fullWidth>
            <InputLabel>Select Area</InputLabel>
            <Select
              value={selectedArea}
              label="Select Area"
              onChange={handleAreaChange}
            >
              <MenuItem value="all">All Areas</MenuItem>
              {trafficData.map((area) => (
                <MenuItem key={area.id} value={area.id}>{area.name}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} md={6}>
          <FormControl fullWidth>
            <InputLabel>Time Range</InputLabel>
            <Select
              value={timeRange}
              label="Time Range"
              onChange={(e) => setTimeRange(e.target.value)}
            >
              <MenuItem value="day">Today</MenuItem>
              <MenuItem value="week">This Week</MenuItem>
              <MenuItem value="month">This Month</MenuItem>
            </Select>
          </FormControl>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Traffic Map */}
        <Grid item xs={12} md={8}>
          <Paper elevation={3} sx={{ p: 2, height: 500 }}>
            <Typography variant="h6" gutterBottom>
              Live Traffic Map
            </Typography>
            <MapContainer 
              center={mapCenter} 
              zoom={mapZoom} 
              style={{ height: "430px", width: "100%" }}
            >
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              />
              
              {trafficData.map((area) => (
                <React.Fragment key={area.id}>
                  <Circle
                    center={[area.lat, area.lng]}
                    radius={500}
                    pathOptions={{
                      color: getCongestionColor(area.congestionLevel),
                      fillColor: getCongestionColor(area.congestionLevel),
                      fillOpacity: 0.5
                    }}
                  />
                  <Marker position={[area.lat, area.lng]}>
                    <Popup>
                      <div>
                        <h3>{area.name}</h3>
                        <p>Congestion Level: {area.congestionLevel}%</p>
                      </div>
                    </Popup>
                  </Marker>
                </React.Fragment>
              ))}
              
              {incidents.map((incident) => (
                <Marker 
                  key={incident.id} 
                  position={[incident.lat, incident.lng]}
                  icon={new L.Icon({
                    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
                    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
                    iconSize: [25, 41],
                    iconAnchor: [12, 41],
                    popupAnchor: [1, -34],
                    shadowSize: [41, 41]
                  })}
                >
                  <Popup>
                    <div>
                      <h3>{incident.type}</h3>
                      <p>Location: {incident.location}</p>
                      <p>Status: {incident.status}</p>
                      <p>Severity: {incident.severity}</p>
                    </div>
                  </Popup>
                </Marker>
              ))}
            </MapContainer>
          </Paper>
        </Grid>

        {/* Traffic Stats */}
        <Grid item xs={12} md={4}>
          <Paper elevation={3} sx={{ p: 2, height: 500, overflow: 'auto' }}>
            <Typography variant="h6" gutterBottom>
              Traffic Incidents
            </Typography>
            {incidents.length === 0 ? (
              <Typography>No current incidents reported.</Typography>
            ) : (
              incidents.map((incident) => (
                <Card key={incident.id} sx={{ mb: 2 }}>
                  <CardContent>
                    <Typography variant="h6" color="text.primary">
                      {incident.type}
                    </Typography>
                    <Typography variant="body2">Location: {incident.location}</Typography>
                    <Typography variant="body2">
                      Status: <span style={{ 
                        color: incident.status === 'Resolved' ? 'green' : 
                               incident.status === 'Clearing' ? 'orange' : 'red' 
                      }}>{incident.status}</span>
                    </Typography>
                    <Typography variant="body2">
                      Severity: {incident.severity}
                    </Typography>
                  </CardContent>
                </Card>
              ))
            )}
          </Paper>
        </Grid>

        {/* Historical Data */}
        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Historical Traffic Patterns (24 Hours)
            </Typography>
            <LineChart
              width={500}
              height={300}
              data={historicalData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="congestion" stroke="#8884d8" activeDot={{ r: 8 }} name="Congestion %" />
              <Line type="monotone" dataKey="incidents" stroke="#82ca9d" name="Incidents" />
            </LineChart>
          </Paper>
        </Grid>

        {/* Predictions */}
        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Traffic Predictions (Next 8 Hours)
            </Typography>
            <BarChart
              width={500}
              height={300}
              data={predictions}
              margin={{ top: 5, right: $0, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="predicted" fill="#8884d8" name="Predicted Congestion %" />
            </BarChart>
          </Paper>
        </Grid>

        {/* Action Buttons */}
        <Grid item xs={12}>
          <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Traffic Management Actions
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
              <Button variant="contained" color="primary">
                Optimize Traffic Signals
              </Button>
              <Button variant="contained" color="secondary">
                Generate Alternate Routes
              </Button>
              <Button variant="contained" color="warning">
                Alert Emergency Services
              </Button>
              <Button variant="contained" color="success">
                Send Public Notifications
              </Button>
              <Button variant="outlined">
                Download Traffic Report
              </Button>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

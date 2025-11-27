# Vision Zero Toronto - Interactive Collision Analysis

An interactive web application for analyzing Toronto's traffic collision data from the [City of Toronto Open Data Portal](https://open.toronto.ca/). This tool provides ward-level collision analysis, filtering capabilities, and statistical insights to support Vision Zero policy effectiveness research.

## Live Demo

Visit the live application: [visionzero.shabanmohammed.com](https://visionzero.shabanmohammed.com)

## Features

- **Interactive Map**: Visualize 18,348 collision records (6,871 unique collisions) from 2006-2023
- **Advanced Filtering**: Filter by severity, road class, time period, weather conditions, and more
- **Ward-Level Analysis**: Drill down to specific Toronto wards and neighbourhoods
- **Statistical Analysis**: Built-in statistical reports with trend analysis and Vision Zero effectiveness metrics
- **Optimized Performance**: Runs efficiently on 512MB RAM deployments
- **Mobile Responsive**: Works on all devices

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ship-shaban/vision-zero.git
cd vision-zero

# Install dependencies
pip install -r requirements.txt

# Optional: Install Parquet support for better performance (recommended)
pip install pyarrow
```

### Running Locally

```bash
python app.py
```

The application will be available at `http://localhost:5001`

On first run, the app will:
1. Process collision data (~8-9 seconds)
2. Create an optimized cache file
3. On subsequent runs, load from cache (~7 seconds)

## Data Sources

This project uses three primary datasets from Toronto Open Data:

1. **Collision Data**: KSI (Killed or Seriously Injured) collision records
   - 18,348 party records (after removing 606 duplicates)
   - Aggregated into 6,871 unique collision events
   - Coverage: 2006-2023

2. **Geographic Boundaries**:
   - 25 City Wards
   - 17 Police Divisions
   - 158 Neighbourhoods

## Project Structure

```
├── app.py                          # Main Flask application
├── generate_enhanced_report.py     # Statistical analysis generator
├── templates/                      # HTML templates
│   ├── index.html                 # Main map interface
│   └── analysis.html              # Statistical analysis page
├── static/                        # CSS, JavaScript, images
├── city-wards/                    # Ward boundary GeoJSON
├── police-divisions/              # Division boundary GeoJSON
├── neighbourhoods/                # Neighbourhood boundary GeoJSON
├── statistical_analysis_results/  # Generated analysis reports
└── requirements.txt               # Python dependencies
```

## Performance Optimizations

This application is optimized to run on low-resource environments (512MB RAM, 0.1 CPU):

### Memory Optimizations
- **60-75% memory reduction**: ~25-30MB runtime vs ~80-100MB baseline
- **Lazy-loaded GeoJSON**: Boundary data loaded only when requested
- **Optimized data types**: Category dtype for categorical columns
- **Column pruning**: Only essential 28 of 60+ columns retained

### Caching Strategy
- **Parquet/Pickle cache**: 67% smaller than CSV (3.2MB vs 9.6MB)
- **Cache versioning**: Automatic invalidation on data updates
- **Fast startup**: 7 seconds with cache vs 8-9 seconds initial processing

### API Optimizations
- **Reduced payload**: 40-60% smaller API responses
- **Compression**: Gzip/Brotli compression enabled
- **Server-side clustering**: Efficient marker aggregation

See [Performance Details](#performance-details) below for technical implementation details.

## API Endpoints

- `GET /` - Main map interface
- `GET /analysis` - Statistical analysis page
- `GET /api/clusters` - Collision markers with server-side clustering
- `GET /api/stats/summary` - Overall collision statistics
- `GET /api/stats/filtered` - Statistics for filtered data
- `GET /api/wards` - Ward boundary GeoJSON (lazy-loaded)
- `GET /api/divisions` - Police division boundaries (lazy-loaded)
- `GET /api/neighbourhoods` - Neighbourhood boundaries (lazy-loaded)

## Deployment

### Render.com (Free Tier)

This application is designed to run on Render's free tier (512MB RAM):

1. Fork this repository
2. Connect your GitHub account to Render
3. Create a new Web Service
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --workers 2 --timeout 120`
   - **Environment**: Python 3.11

### Other Platforms

Works on any platform supporting Python Flask:
- Heroku
- Railway
- DigitalOcean App Platform
- AWS Elastic Beanstalk
- Google Cloud Run

## Contributing

Contributions are welcome! This project is open source and welcomes contributions from the community!

**Repository**: https://github.com/ship-shaban/vision-zero

### Areas for Contribution

- **Features**: Additional filtering options, export capabilities, data visualizations
- **Analysis**: New statistical methods, trend detection, predictive modeling
- **Performance**: Further optimizations, caching strategies
- **Documentation**: Tutorials, API documentation, deployment guides
- **Accessibility**: Screen reader support, keyboard navigation
- **Internationalization**: Multi-language support

### Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pyarrow  # Optional but recommended

# Run the application
python app.py

# Run validation scripts
python validate_implementation.py
python validate_against_police_dashboard.py
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test locally
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your fork (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

MIT License - See LICENSE file for details

## Data Attribution

Collision data provided by the City of Toronto under the [Open Government License - Toronto](https://open.toronto.ca/open-data-license/)

## Acknowledgments

- City of Toronto Open Data Portal
- Toronto Police Service for collision data validation
- Flask, Pandas, and Shapely communities

---

## Performance Details

### Implemented Optimizations

#### Phase 1: Critical Memory Optimizations

**1. Ward Assignment Caching**
- **Problem**: Point-in-polygon calculations ran on every startup
- **Solution**: Execute only when cache is stale or missing
- **Impact**: 70-80% faster startup

**2. Lazy-Load GeoJSON**
- **Problem**: 4MB of boundary data loaded at startup
- **Solution**: Load only when API endpoints are requested
- **Impact**: 20-30MB memory reduction

**3. Parquet/Pickle Cache**
- **Problem**: JSON cache was large and slow
- **Solution**: Binary format with automatic fallback
- **Impact**: 67% smaller, 18% faster load

**4. Reduced Marker Payload**
- **Problem**: Unnecessary fields in API responses
- **Solution**: Removed unused environmental and factor fields
- **Impact**: 40-60% smaller responses

#### Phase 2: Data Type Optimizations

**5. DataFrame Column Pruning**
- **Problem**: 60+ columns when only 28 needed
- **Solution**: Drop unused columns immediately after CSV load
- **Impact**: 40% DataFrame memory reduction

**6. Category Dtype Conversion**
- **Problem**: String dtype for low-cardinality columns
- **Solution**: Convert to category dtype (integer codes + lookup)
- **Impact**: 50-90% memory reduction per column

### Performance Comparison

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Startup memory | 80-100MB | 25-30MB | 60-75% |
| Cached startup | 15s | 7s | 53% faster |
| Cache file size | 12MB | 3.2MB | 73% smaller |
| API response | 200KB | 80KB | 60% smaller |
| GeoJSON loading | At startup | On-demand | Lazy |

### Memory Profiling

```bash
# Test startup time
time python -c "import app"

# Check cache size
ls -lh processed_collisions_cache_v*.pkl

# Monitor memory usage
import psutil, os
process = psutil.Process(os.getpid())
print(f"Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

---

**Last Updated**: November 2025
**Python**: 3.11+
**Status**: Production Ready

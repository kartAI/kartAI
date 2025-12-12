# Workshop Data Sources: WMS/PostgreSQL vs COG/FlatGeobuf

## Comparison Overview

| Aspect | Current (WMS/PostgreSQL) | Proposed (COG/FlatGeobuf) |
|--------|-------------------------|---------------------------|
| **Setup Complexity** | Requires API keys and database credentials | No credentials if files are public |
| **Dependencies** | Active WMS service, PostgreSQL database | Static files on any hosting |
| **Performance** | Network latency for each request | Direct file access, HTTP caching |
| **Offline Use** | Not possible | Possible with local files |
| **Scalability** | Limited by service capacity | CDN-friendly, unlimited read scaling |
| **Cost** | Server/database costs | Storage + bandwidth costs |
| **Workshop Prep** | Generate credentials for each workshop | Upload files once, reuse forever |
| **Data Consistency** | Live data may change | Frozen snapshot, consistent results |

## Current Workshop Flow (WMS/PostgreSQL)

### Required Secrets
```python
os.environ['NK_WMS_API_KEY'] = "xxx"  # WMS API key
os.environ['OSM_DB_PWD'] = "yyy"      # PostgreSQL password
```

### Configuration
```json
{
  "ImageSources": [
    {
      "name": "OrtofotoWMS",
      "type": "WMSImageSource",
      "url": "https://waapi.webatlas.no/wms-orto/",
      "api_key": "NK_WMS_API_KEY",
      "layers": ["ortofoto"]
    },
    {
      "name": "OSMByggDb",
      "type": "PostgresImageSource",
      "host": "postgresql-dev-kartai.postgres.database.azure.com",
      "database": "kartai_opendata",
      "user": "kartai_opendata_ro@postgresql-dev-kartai",
      "passwd": "OSM_DB_PWD",
      "table": "public.osm_buildings"
    }
  ]
}
```

### Data Flow
1. Workshop participant requests secrets from organizers
2. Notebook connects to WMS service for each image tile
3. Notebook connects to PostgreSQL for building geometries
4. Data is fetched over network during training
5. Cached locally after first fetch

## Proposed Workshop Flow (COG/FlatGeobuf)

### Required Secrets
```python
# No secrets required if files are publicly accessible!
# Or optionally, a single SAS token for Azure Blob Storage
```

### Configuration
```json
{
  "ImageSources": [
    {
      "name": "OrtofotoFile",
      "type": "ImageFileImageSource",
      "file_path": "https://storage.example.com/workshop/sand_ortofoto.tif"
    },
    {
      "name": "OSMByggFile",
      "type": "VectorFileImageSource",
      "file_path": "https://storage.example.com/workshop/sand_buildings.fgb",
      "srid": 25832
    }
  ]
}
```

### Data Flow
1. Workshop participant runs notebook (no secrets needed)
2. Notebook downloads COG file once (with HTTP range requests for tiles)
3. Notebook downloads FlatGeobuf once (with spatial indexing)
4. Files are cached locally
5. Subsequent access uses cached files

## Benefits Breakdown

### 1. Simplified Setup
**Before**: Organizers must:
- Generate WMS API key with expiration
- Provide read-only database credentials
- Share secrets via GitHub Gist
- Manage credential lifecycle

**After**: Organizers must:
- Upload files to public storage once
- Share notebook with file URLs
- No credential management needed

### 2. Better Performance
**WMS**:
- Each tile requires separate HTTP request
- No built-in caching at protocol level
- Must include authentication in each request

**COG**:
- HTTP range requests for efficient partial reads
- Browser/CDN caching works naturally
- Single file with internal tiling

**PostgreSQL**:
- Query execution time varies
- Connection overhead for each request
- Requires database server availability

**FlatGeobuf**:
- Spatial index allows efficient bbox queries
- Simple file download or streaming
- No server-side computation needed

### 3. Workshop Reliability
**Current challenges**:
- WMS service downtime affects all workshops
- Database connection limits (max concurrent users)
- API key expiration requires regeneration
- Network issues affect data access

**With file-based approach**:
- Files are static and immutable
- CDN provides global distribution
- No connection limits
- Can work offline after initial download

### 4. Cost Efficiency
**Current costs**:
- WMS service hosting and bandwidth
- PostgreSQL database hosting
- Credential management overhead

**File-based costs**:
- One-time storage cost (minimal)
- Bandwidth costs (can use free tiers with CDN)
- No ongoing service costs

## Implementation Path

### Phase 1: Data Preparation (Current State)
- ✅ Created config template
- ✅ Documented setup process  
- ✅ Created data preparation script
- ⏳ Need decisions on hosting and data preparation

### Phase 2: Data Creation & Hosting
- [ ] Run `prepare_workshop_data.py` to create files
- [ ] Upload to chosen hosting (Azure Blob recommended)
- [ ] Verify file accessibility and performance
- [ ] Create production config with actual URLs

### Phase 3: Notebook Update
- [ ] Update config reference in notebook
- [ ] Remove WMS/PostgreSQL credential requirements
- [ ] Add file download progress indicators (optional)
- [ ] Test complete workshop flow

### Phase 4: Documentation & Testing
- [ ] Update workshop README
- [ ] Create troubleshooting guide
- [ ] Run full workshop dry-run
- [ ] Gather feedback and iterate

## Questions to Answer

1. **Hosting location**: Where should workshop files be stored?
   - Azure Blob Storage (same as existing infrastructure)
   - AWS S3
   - GitHub releases (size limited)
   - Other CDN service

2. **Data scope**: Which areas to support?
   - Current 3 areas (Sand, Skøyen, Midtbyen)
   - Subset of areas
   - Additional areas

3. **Data preparation**: Who prepares the data?
   - Run provided script with existing credentials
   - Use existing files if available
   - Create new datasets

4. **Access control**: Should files be:
   - Fully public (easiest for participants)
   - SAS token protected (requires sharing token)
   - Authenticated (more complex setup)

## Recommendation

**Recommended approach:**
1. Use Azure Blob Storage (consistent with existing infrastructure)
2. Make files public or use long-lived SAS tokens
3. Support all 3 current workshop areas
4. Use provided script to generate files from existing sources

**Expected file sizes:**
- Sand area: ~150 MB (COG) + ~5 MB (FlatGeobuf) = ~155 MB
- Skøyen area: ~50 MB (COG) + ~3 MB (FlatGeobuf) = ~53 MB
- Midtbyen area: ~60 MB (COG) + ~4 MB (FlatGeobuf) = ~64 MB
- **Total: ~272 MB** for all areas

This is manageable for cloud storage and reasonable for workshop participants to download.

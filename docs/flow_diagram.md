# RAP System Flow Diagrams

## 1. Main Processing Flow

```mermaid
flowchart TD
    A[Start] --> B[Initialize VideoProcessor]
    B --> C[Load Models: YOLOv5, Haar Cascade]
    C --> D[Open Video Source]
    D --> E{Video Source Valid?}
    E -->|No| F[Error: Cannot open video]
    E -->|Yes| G[Read Frame]
    G --> H{Frame Valid?}
    H -->|No| I[End of Video]
    H -->|Yes| J[Increment Frame Counter]
    J --> K[Detect Persons with YOLOv5]
    K --> L[For Each Person Detection]
    L --> M[Detect Faces with Haar Cascade]
    M --> N[Extract Face Encodings]
    N --> O[Re-identify Persons]
    O --> P[Update Tracking]
    P --> Q[Visualize Results]
    Q --> R[Export Data Every 30 Frames]
    R --> S{User Input?}
    S -->|'q'| T[Save and Exit]
    S -->|'s'| U[Save Current State]
    S -->|None| G
    U --> G
    T --> V[End]
    I --> T
```

## 2. Person Detection Flow

```mermaid
flowchart TD
    A[Input Frame] --> B[YOLOv5 Inference]
    B --> C[Filter Person Class]
    C --> D[Apply Confidence Threshold]
    D --> E[Filter by Size Constraints]
    E --> F[Extract Bounding Boxes]
    F --> G[Calculate Centers]
    G --> H[Return Detections]
    
    style A fill:#e1f5fe
    style H fill:#c8e6c9
```

## 3. Face Detection and Recognition Flow

```mermaid
flowchart TD
    A[Person Bounding Box] --> B[Extract Person ROI]
    B --> C[Convert to Grayscale]
    C --> D[Haar Cascade Detection]
    D --> E[For Each Face]
    E --> F[Extract Face ROI]
    F --> G[Convert to RGB]
    G --> H[Face Encoding Extraction]
    H --> I{Encoding Successful?}
    I -->|No| J[Skip Face]
    I -->|Yes| K[Compare with Known Faces]
    K --> L{Match Found?}
    L -->|Yes| M[Update Person ID]
    L -->|No| N[Add to Known Faces]
    M --> O[Return Face Data]
    N --> O
    J --> O
    
    style A fill:#e1f5fe
    style O fill:#c8e6c9
```

## 4. Multi-Object Tracking Flow

```mermaid
flowchart TD
    A[Current Detections] --> B[For Each Tracked Person]
    B --> C[Calculate IOU Scores]
    C --> D[Calculate Distance Scores]
    D --> E[Combine Scores]
    E --> F{Best Match Found?}
    F -->|Yes| G[Update Track]
    F -->|No| H[Increment Disappeared Count]
    H --> I{Max Disappeared?}
    I -->|No| J[Keep Track]
    I -->|Yes| K[Remove Track]
    G --> L[Mark Detection as Used]
    J --> L
    K --> L
    L --> M[For Unused Detections]
    M --> N[Create New Track]
    N --> O[Return Updated Tracks]
    
    style A fill:#e1f5fe
    style O fill:#c8e6c9
```

## 5. Face Re-identification Flow

```mermaid
flowchart TD
    A[Face Encoding] --> B{Encoding Valid?}
    B -->|No| C[Return No Match]
    B -->|Yes| D[For Each Known Face]
    D --> E{Known Face Valid?}
    E -->|No| F[Skip]
    E -->|Yes| G[Calculate Similarity]
    G --> H{Similarity > Threshold?}
    H -->|No| F
    H -->|Yes| I{Best Match So Far?}
    I -->|No| F
    I -->|Yes| J[Update Best Match]
    F --> K{More Known Faces?}
    K -->|Yes| D
    K -->|No| L[Return Best Match]
    J --> K
    
    style A fill:#e1f5fe
    style L fill:#c8e6c9
    style C fill:#ffcdd2
```

## 6. Data Export Flow

```mermaid
flowchart TD
    A[Frame Processing Complete] --> B{Export Interval?}
    B -->|No| C[Continue]
    B -->|Yes| D[Save Frame Image]
    D --> E[For Each Active Person]
    E --> F[Save Face Images]
    F --> G[Create Detection Entry]
    G --> H[Add to Detection Log]
    H --> I[Update Face Encodings]
    I --> J[Continue]
    
    style A fill:#e1f5fe
    style J fill:#c8e6c9
```

## 7. Visualization Flow

```mermaid
flowchart TD
    A[Tracked People] --> B[For Each Person]
    B --> C[Draw Bounding Box]
    C --> D{Person Disappeared?}
    D -->|Yes| E[Use Orange Color]
    D -->|No| F[Use Assigned Color]
    E --> G[Draw Tracking Trail]
    F --> G
    G --> H[For Each Face]
    H --> I[Draw Face Box]
    I --> J[Add Labels]
    J --> K[Display Statistics]
    K --> L[Show Frame]
    
    style A fill:#e1f5fe
    style L fill:#c8e6c9
```

## 8. Error Handling Flow

```mermaid
flowchart TD
    A[Operation] --> B{Operation Successful?}
    B -->|Yes| C[Continue Processing]
    B -->|No| D[Log Error]
    D --> E{Error Type?}
    E -->|Critical| F[Stop Processing]
    E -->|Non-Critical| G[Skip Operation]
    E -->|Recoverable| H[Retry Operation]
    G --> I[Continue with Next]
    H --> J{Retry Successful?}
    J -->|Yes| C
    J -->|No| G
    F --> K[Cleanup and Exit]
    
    style A fill:#e1f5fe
    style C fill:#c8e6c9
    style K fill:#ffcdd2
```

## 9. Memory Management Flow

```mermaid
flowchart TD
    A[Periodic Cleanup] --> B[Clean Known Faces]
    B --> C[Remove None Values]
    C --> D[Trim Tracking History]
    D --> E[Remove Old Tracks]
    E --> F[Garbage Collection]
    F --> G[Continue Processing]
    
    style A fill:#e1f5fe
    style G fill:#c8e6c9
```

## 10. User Interaction Flow

```mermaid
flowchart TD
    A[Wait for Key Press] --> B{Key Pressed?}
    B -->|No| C[Continue Processing]
    B -->|Yes| D{Key Type?}
    D -->|'q'| E[Save Detection Log]
    D -->|'s'| F[Save Current State]
    D -->|Other| C
    E --> G[Save Face Encodings]
    G --> H[Close Video]
    H --> I[Exit Program]
    F --> J[Print Save Message]
    J --> C
    
    style A fill:#e1f5fe
    style C fill:#c8e6c9
    style I fill:#ffcdd2
```

## Key Decision Points

### 1. Detection Thresholds
- **Person Detection**: Confidence > 0.5
- **Face Recognition**: Similarity > 0.6
- **Tracking**: IOU > 0.3

### 2. Timing Intervals
- **Export**: Every 30 frames
- **Cleanup**: Every 100 frames
- **Display**: Every frame

### 3. Memory Limits
- **Tracking History**: 30 frames per person
- **Disappeared Count**: 30 frames maximum
- **Color Palette**: 100 unique colors

### 4. Performance Optimizations
- **Batch Processing**: Face encodings
- **Early Termination**: Invalid detections
- **Memory Pooling**: Reusable objects
- **GPU Acceleration**: YOLOv5 inference 
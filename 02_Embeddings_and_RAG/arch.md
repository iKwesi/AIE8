# Enhanced RAG System Architecture

## Clean Architecture Overview

```mermaid
flowchart LR
    %% Input Section
    subgraph INPUT [" "]
        Q[🔍<br/>What is the Michael<br/>Eisner Memorial Weak<br/>Executive Problem?]
    end

    %% OpenAI API Section
    subgraph API ["OpenAI API"]
        EM[🧠<br/>Embedding Model<br/>text-embedding-3-small]
        CM[💬<br/>Chat Model<br/>GPT-4o-mini]
    end

    %% Prompt Templates Section
    subgraph PROMPT ["Prompt Templates"]
        SYS[📝 System Prompt<br/>Use the provided context to answer the user's query.<br/><br/>You may not answer the user's query unless there is specific<br/>context in the following text.<br/><br/>If you do not know the answer, or cannot answer, please respond<br/>with "I don't know".]
        
        CTX[📋 Context:<br/>{context}]
        
        USER[👤 User Query:<br/>{user_query}]
    end

    %% Vector Store Section
    subgraph VECTOR ["Vector Store"]
        APP1[📱<br/>APP<br/>App Logic]
        VDB[(🗄️<br/>Vector Database)]
        APP2[📱<br/>APP<br/>App Logic]
        
        SEARCH[🔍 Find Nearest<br/>Neighbours<br/>cosine similarity]
        RETURN[📤 Return document(s)<br/>from<br/>Nearest Neighbours]
    end

    %% Output Section
    subgraph OUTPUT [" "]
        RESP[🔍<br/><br/>The Michael<br/>Eisner...]
    end

    %% Connections
    Q --> EM
    EM --> |[0.1, 0.4, -0.6, ...]| VECTOR
    
    APP1 --> SEARCH
    SEARCH --> VDB
    VDB --> RETURN
    RETURN --> APP2
    
    APP2 --> CTX
    CTX --> CM
    USER --> CM
    SYS --> CM
    
    CM --> RESP

    %% Styling
    classDef inputStyle fill:#f8f9fa,stroke:#6c757d,stroke-width:2px,color:#000
    classDef apiStyle fill:#17a2b8,stroke:#138496,stroke-width:2px,color:#fff
    classDef promptStyle fill:#ffc107,stroke:#e0a800,stroke-width:2px,color:#000
    classDef vectorStyle fill:#6f42c1,stroke:#5a32a3,stroke-width:2px,color:#fff
    classDef outputStyle fill:#f8f9fa,stroke:#6c757d,stroke-width:2px,color:#000

    class Q,RESP inputStyle
    class EM,CM apiStyle
    class SYS,CTX,USER promptStyle
    class APP1,VDB,APP2,SEARCH,RETURN vectorStyle
```

## Enhanced Features Architecture

```mermaid
flowchart TB
    %% Input Sources
    subgraph SOURCES ["📥 Enhanced Input Sources"]
        TXT[📄 Text Files<br/>PMarca Blogs]
        PDF[📕 PDF Documents<br/>Page Attribution]
        YT[📺 YouTube Videos<br/>Timestamp Metadata]
    end

    %% Processing Layer
    subgraph PROCESS ["🔄 Document Processing"]
        LOAD[Document Loaders<br/>TextFileLoader | PDFFileLoader | YouTubeLoader]
        SPLIT[Text Splitters<br/>Character | PDF | YouTube]
        META[Metadata Extraction<br/>Pages | Timestamps | Categories]
    end

    %% Embedding Layer
    subgraph EMBED ["🧠 Multiple Embedding Models"]
        E1[text-embedding-3-small<br/>1536 dimensions]
        E2[text-embedding-3-large<br/>3072 dimensions]
        E3[text-embedding-ada-002<br/>1536 dimensions]
    end

    %% Enhanced Vector Database
    subgraph ENHANCED ["🗄️ Enhanced Vector Database"]
        subgraph METRICS ["📊 Distance Metrics"]
            COS[Cosine]
            EUC[Euclidean]
            MAN[Manhattan]
            DOT[Dot Product]
        end
        
        VDB[(Vector Database<br/>with Metadata)]
        
        FILTER[🎯 Metadata Filtering<br/>Category | Importance | Source]
    end

    %% RAG Pipeline
    subgraph RAG ["🤖 Enhanced RAG Pipeline"]
        QUERY[User Query]
        SEARCH[Semantic Search<br/>+ Metadata Filtering]
        CONTEXT[Context Assembly<br/>+ Source Attribution]
        PROMPT[Enhanced Prompting<br/>System + User Templates]
        LLM[ChatOpenAI<br/>GPT-4o-mini]
        RESPONSE[Response + Citations<br/>+ Similarity Scores]
    end

    %% Data Flow
    SOURCES --> PROCESS
    PROCESS --> EMBED
    EMBED --> ENHANCED
    ENHANCED --> RAG
    
    QUERY --> SEARCH
    SEARCH --> CONTEXT
    CONTEXT --> PROMPT
    PROMPT --> LLM
    LLM --> RESPONSE

    %% Styling
    classDef sourceStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef processStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef embedStyle fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef enhancedStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef ragStyle fill:#fce4ec,stroke:#c2185b,stroke-width:2px

    class TXT,PDF,YT sourceStyle
    class LOAD,SPLIT,META processStyle
    class E1,E2,E3 embedStyle
    class COS,EUC,MAN,DOT,VDB,FILTER enhancedStyle
    class QUERY,SEARCH,CONTEXT,PROMPT,LLM,RESPONSE ragStyle
```

## Key Enhancement Summary

### 🎯 **5 Major Enhancements Implemented**

1. **📕 PDF Processing**
   - Page-level metadata extraction
   - Source attribution with page numbers
   - Rich document metadata (title, author, creation date)

2. **📊 Multiple Distance Metrics**
   - Cosine Similarity (default)
   - Euclidean Distance
   - Manhattan Distance  
   - Dot Product

3. **🏷️ Enhanced Metadata Support**
   - Rich metadata storage and filtering
   - Category-based organization
   - Importance level classification
   - Source type tracking

4. **🧠 Multiple Embedding Models**
   - text-embedding-3-small (1536D, cost-effective)
   - text-embedding-3-large (3072D, highest quality)
   - text-embedding-ada-002 (1536D, legacy support)

5. **📺 YouTube Integration**
   - Transcript extraction from video URLs
   - Timestamp preservation in chunks
   - Video metadata (title, duration, views)

### 🔍 **Enhanced Query Processing**

- **Semantic Search**: Multiple distance metrics for better relevance
- **Metadata Filtering**: Target specific document types or categories
- **Source Attribution**: Full traceability with page numbers and timestamps
- **Quality Controls**: "I don't know" responses for out-of-domain queries

### 📊 **Performance & Analytics**

- **Similarity Scoring**: Transparent relevance metrics
- **Database Statistics**: Performance monitoring and insights
- **Response Analytics**: Context count and source breakdown
- **Knowledge Boundaries**: Proper handling of irrelevant queries

## Technical Specifications

| Component | Details |
|-----------|---------|
| **Input Formats** | Text (.txt), PDF (.pdf), YouTube URLs |
| **Embedding Models** | OpenAI text-embedding-3-small/large, ada-002 |
| **Vector Dimensions** | 1536 (small/ada-002), 3072 (large) |
| **Distance Metrics** | Cosine, Euclidean, Manhattan, Dot Product |
| **Metadata Fields** | Source, Category, Importance, Pages, Timestamps |
| **LLM Model** | GPT-4o-mini via OpenAI API |
| **Response Features** | Citations, Similarity Scores, Source Attribution |

This enhanced RAG system provides a production-ready, multi-modal document processing and retrieval solution with comprehensive quality controls and transparency features.

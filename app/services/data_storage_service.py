from sqlalchemy.orm import Session, Query
from sqlalchemy import or_, and_, func, text
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import redis
import elasticsearch
from elasticsearch import Elasticsearch
from fastapi import HTTPException
import pandas as pd

class EnhancedDataStorageService:
    def __init__(self, db: Session):
        self.db = db
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        self.es_client = Elasticsearch(['http://localhost:9200']) if self._check_elasticsearch() else None
    
    def _check_elasticsearch(self) -> bool:
        """Check if Elasticsearch is available"""
        try:
            es = Elasticsearch(['http://localhost:9200'])
            return es.ping()
        except:
            return False
    
    async def advanced_search(
        self, 
        model_class,
        search_params: Dict[str, Any],
        page: int = 1,
        page_size: int = 50,
        sort_by: str = None,
        sort_order: str = 'asc'
    ) -> Dict[str, Any]:
        """Advanced search with filtering, sorting, and pagination"""
        
        query = self.db.query(model_class)
        
        # Apply filters
        query = self._apply_filters(query, model_class, search_params.get('filters', {}))
        
        # Apply full-text search if search term provided
        if search_params.get('search_term'):
            query = self._apply_full_text_search(query, model_class, search_params['search_term'])
        
        # Apply date range filters
        if search_params.get('date_range'):
            query = self._apply_date_range(query, model_class, search_params['date_range'])
        
        # Get total count before pagination
        total_count = query.count()
        
        # Apply sorting
        if sort_by:
            sort_column = getattr(model_class, sort_by, None)
            if sort_column:
                if sort_order.lower() == 'desc':
                    query = query.order_by(sort_column.desc())
                else:
                    query = query.order_by(sort_column.asc())
        
        # Apply pagination
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        results = query.all()
        
        return {
            'data': results,
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total_count': total_count,
                'total_pages': (total_count + page_size - 1) // page_size,
                'has_next': page * page_size < total_count,
                'has_previous': page > 1
            },
            'search_metadata': {
                'applied_filters': search_params.get('filters', {}),
                'sorting': {'field': sort_by, 'order': sort_order},
                'search_term': search_params.get('search_term')
            }
        }
    
    def _apply_filters(self, query: Query, model_class, filters: Dict[str, Any]) -> Query:
        """Apply various filter types to query"""
        
        for field, filter_value in filters.items():
            if hasattr(model_class, field):
                column = getattr(model_class, field)
                
                if isinstance(filter_value, dict):
                    # Complex filter operations
                    if 'eq' in filter_value:
                        query = query.filter(column == filter_value['eq'])
                    elif 'ne' in filter_value:
                        query = query.filter(column != filter_value['ne'])
                    elif 'gt' in filter_value:
                        query = query.filter(column > filter_value['gt'])
                    elif 'gte' in filter_value:
                        query = query.filter(column >= filter_value['gte'])
                    elif 'lt' in filter_value:
                        query = query.filter(column < filter_value['lt'])
                    elif 'lte' in filter_value:
                        query = query.filter(column <= filter_value['lte'])
                    elif 'in' in filter_value:
                        query = query.filter(column.in_(filter_value['in']))
                    elif 'like' in filter_value:
                        query = query.filter(column.like(f"%{filter_value['like']}"))
                    elif 'ilike' in filter_value:
                        query = query.filter(column.ilike(f"%{filter_value['ilike']}"))
                else:
                    # Simple equality filter
                    query = query.filter(column == filter_value)
        
        return query
    
    def _apply_full_text_search(self, query: Query, model_class, search_term: str) -> Query:
        """Apply full-text search across multiple fields"""
        
        search_filters = []
        searchable_fields = self._get_searchable_fields(model_class)
        
        for field in searchable_fields:
            column = getattr(model_class, field)
            search_filters.append(column.ilike(f"%{search_term}%"))
        
        if search_filters:
            query = query.filter(or_(*search_filters))
        
        return query
    
    def _apply_date_range(self, query: Query, model_class, date_range: Dict[str, str]) -> Query:
        """Apply date range filters"""
        
        date_filters = []
        
        for field, range_config in date_range.items():
            if hasattr(model_class, field):
                column = getattr(model_class, field)
                
                if 'start' in range_config:
                    start_date = datetime.fromisoformat(range_config['start'].replace('Z', '+00:00'))
                    date_filters.append(column >= start_date)
                
                if 'end' in range_config:
                    end_date = datetime.fromisoformat(range_config['end'].replace('Z', '+00:00'))
                    date_filters.append(column <= end_date)
        
        if date_filters:
            query = query.filter(and_(*date_filters))
        
        return query
    
    def _get_searchable_fields(self, model_class) -> List[str]:
        """Get list of searchable fields for a model"""
        # This could be enhanced with model metadata or configuration
        text_fields = ['name', 'description', 'title', 'customer_name', 'institution_name']
        return [field for field in text_fields if hasattr(model_class, field)]
    
    async def elasticsearch_search(self, index: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Perform advanced search using Elasticsearch"""
        
        if not self.es_client:
            raise HTTPException(status_code=503, detail="Elasticsearch service unavailable")
        
        try:
            response = self.es_client.search(index=index, body=query)
            
            hits = response['hits']['hits']
            total = response['hits']['total']['value']
            
            return {
                'results': [hit['_source'] for hit in hits],
                'total': total,
                'took': response['took'],
                'timed_out': response['timed_out']
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    async def cache_data(self, key: str, data: Any, expire_seconds: int = 3600) -> bool:
        """Cache data in Redis"""
        try:
            serialized_data = json.dumps(data, default=str)
            self.redis_client.setex(key, expire_seconds, serialized_data)
            return True
        except Exception as e:
            print(f"Cache set failed: {e}")
            return False
    
    async def get_cached_data(self, key: str) -> Optional[Any]:
        """Retrieve cached data from Redis"""
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            print(f"Cache get failed: {e}")
            return None
    
    async def invalidate_cache(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            print(f"Cache invalidation failed: {e}")
            return 0
    
    async def get_data_export(
        self, 
        model_class,
        filters: Dict[str, Any],
        format: str = 'excel',
        columns: List[str] = None
    ) -> Dict[str, Any]:
        """Export data in various formats"""
        
        query = self.db.query(model_class)
        query = self._apply_filters(query, model_class, filters)
        results = query.all()
        
        # Convert to DataFrame
        data = []
        for result in results:
            row = {}
            for column in model_class.__table__.columns:
                if not columns or column.name in columns:
                    row[column.name] = getattr(result, column.name)
            data.append(row)
        
        df = pd.DataFrame(data)
        
        if format == 'excel':
            return await self._export_excel(df)
        elif format == 'csv':
            return await self._export_csv(df)
        elif format == 'json':
            return await self._export_json(df)
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
    
    async def _export_excel(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Export DataFrame to Excel"""
        from io import BytesIO
        import tempfile
        import os
        
        try:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            
            output.seek(0)
            filename = f"export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            return {
                'filename': filename,
                'content': output.getvalue(),
                'content_type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Excel export failed: {str(e)}")
    
    async def _export_csv(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Export DataFrame to CSV"""
        from io import StringIO
        
        try:
            output = StringIO()
            df.to_csv(output, index=False)
            csv_content = output.getvalue()
            output.close()
            
            filename = f"export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
            
            return {
                'filename': filename,
                'content': csv_content,
                'content_type': 'text/csv'
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"CSV export failed: {str(e)}")
    
    async def _export_json(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Export DataFrame to JSON"""
        try:
            json_content = df.to_json(orient='records', date_format='iso')
            filename = f"export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            
            return {
                'filename': filename,
                'content': json_content,
                'content_type': 'application/json'
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"JSON export failed: {str(e)}")
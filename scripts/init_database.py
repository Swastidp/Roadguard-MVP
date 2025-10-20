"""
Database initialization script for RoadGuard.

This script creates the SQLite database schema and optionally populates it
with sample hazard data for testing and demonstration.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
import random
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import DB_PATH, CLASS_NAMES


# ============================================================================
# Database Schema
# ============================================================================

def create_tables(conn: sqlite3.Connection) -> None:
    """
    Create database tables and indexes.
    
    Args:
        conn: SQLite database connection
        
    Raises:
        sqlite3.Error: If table creation fails
    """
    cursor = conn.cursor()
    
    try:
        print("📊 Creating database tables...")
        
        # Main hazards table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hazards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                class_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                severity TEXT NOT NULL CHECK(severity IN ('critical', 'high', 'medium', 'low')),
                confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                bbox TEXT,
                image_path TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active' CHECK(status IN ('active', 'resolved', 'pending', 'verified')),
                report_count INTEGER DEFAULT 1 CHECK(report_count >= 1),
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        print("  ✅ Created 'hazards' table")
        
        # Create indexes for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_hazards_location 
            ON hazards(latitude, longitude)
        """)
        print("  ✅ Created index on location")
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_hazards_timestamp 
            ON hazards(timestamp)
        """)
        print("  ✅ Created index on timestamp")
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_hazards_status 
            ON hazards(status)
        """)
        print("  ✅ Created index on status")
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_hazards_severity 
            ON hazards(severity)
        """)
        print("  ✅ Created index on severity")
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_hazards_class 
            ON hazards(class_name)
        """)
        print("  ✅ Created index on class_name")
        
        # Create trigger to update updated_at timestamp
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS update_hazards_timestamp 
            AFTER UPDATE ON hazards
            BEGIN
                UPDATE hazards SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
        """)
        print("  ✅ Created update trigger")
        
        conn.commit()
        print("✅ Database schema created successfully!\n")
        
    except sqlite3.Error as e:
        print(f"❌ Error creating tables: {e}")
        raise


# ============================================================================
# Sample Data Generation
# ============================================================================

def generate_sample_hazards() -> list:
    """
    Generate sample hazard data for testing.
    
    Returns:
        List of hazard dictionaries with realistic data
    """
    
    # Major city coordinates (Delhi, Mumbai, Bangalore, Kolkata)
    city_centers = {
        'Delhi': (28.6139, 77.2090),
        'Mumbai': (19.0760, 72.8777),
        'Bangalore': (12.9716, 77.5946),
        'Kolkata': (22.5726, 88.3639)
    }
    
    # Road offsets for realistic positioning
    road_patterns = [
        (0.001, 0.001),   # Northeast
        (-0.001, 0.001),  # Southeast
        (-0.001, -0.001), # Southwest
        (0.001, -0.001),  # Northwest
        (0.002, 0),       # East
        (-0.002, 0),      # West
        (0, 0.002),       # North
        (0, -0.002),      # South
    ]
    
    hazards = []
    
    # Generate hazards for each city
    for city, (base_lat, base_lon) in city_centers.items():
        num_hazards = random.randint(5, 8)
        
        for i in range(num_hazards):
            # Apply random offset to simulate road locations
            lat_offset, lon_offset = random.choice(road_patterns)
            lat_offset += random.uniform(-0.0005, 0.0005)
            lon_offset += random.uniform(-0.0005, 0.0005)
            
            latitude = base_lat + lat_offset
            longitude = base_lon + lon_offset
            
            # Random class
            class_id = random.randint(0, len(CLASS_NAMES) - 1)
            class_name = CLASS_NAMES[class_id]
            
            # Severity based on class and random chance
            if class_name == 'pothole':
                severity = random.choices(
                    ['critical', 'high', 'medium', 'low'],
                    weights=[0.3, 0.4, 0.2, 0.1]
                )[0]
            elif class_name == 'alligator_crack':
                severity = random.choices(
                    ['critical', 'high', 'medium', 'low'],
                    weights=[0.2, 0.5, 0.2, 0.1]
                )[0]
            else:
                severity = random.choices(
                    ['critical', 'high', 'medium', 'low'],
                    weights=[0.1, 0.2, 0.4, 0.3]
                )[0]
            
            # Confidence score (higher for clearer hazards)
            if severity in ['critical', 'high']:
                confidence = random.uniform(0.75, 0.95)
            else:
                confidence = random.uniform(0.55, 0.85)
            
            # Bounding box (normalized coordinates)
            bbox = [
                random.uniform(0.2, 0.6),  # x1
                random.uniform(0.3, 0.7),  # y1
                random.uniform(0.4, 0.8),  # x2
                random.uniform(0.5, 0.9)   # y2
            ]
            bbox_json = json.dumps(bbox)
            
            # Timestamp (last 30 days)
            days_ago = random.randint(0, 30)
            hours_ago = random.randint(0, 23)
            timestamp = datetime.now() - timedelta(days=days_ago, hours=hours_ago)
            
            # Status (mostly active)
            status = random.choices(
                ['active', 'pending', 'resolved'],
                weights=[0.7, 0.2, 0.1]
            )[0]
            
            # Report count (active hazards may have multiple reports)
            if status == 'active' and severity in ['critical', 'high']:
                report_count = random.randint(1, 5)
            else:
                report_count = 1
            
            # Last seen (same as timestamp for new, or more recent for reported)
            if report_count > 1:
                last_seen = timestamp + timedelta(days=random.randint(1, 7))
            else:
                last_seen = timestamp
            
            hazard = {
                'latitude': round(latitude, 6),
                'longitude': round(longitude, 6),
                'class_id': class_id,
                'class_name': class_name,
                'severity': severity,
                'confidence': round(confidence, 3),
                'bbox': bbox_json,
                'image_path': f"data/images/{city.lower()}_hazard_{i+1}.jpg",
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'status': status,
                'report_count': report_count,
                'last_seen': last_seen.strftime('%Y-%m-%d %H:%M:%S'),
                'notes': f"Sample {class_name.replace('_', ' ')} detected in {city}"
            }
            
            hazards.append(hazard)
    
    return hazards


def insert_sample_hazards(conn: sqlite3.Connection) -> None:
    """
    Insert sample hazard data into the database.
    
    Args:
        conn: SQLite database connection
        
    Raises:
        sqlite3.Error: If insertion fails
    """
    cursor = conn.cursor()
    
    try:
        print("🎲 Generating sample hazard data...")
        
        # Generate sample hazards
        hazards = generate_sample_hazards()
        
        print(f"  📊 Generated {len(hazards)} sample hazards")
        
        # Insert hazards using parameterized queries (prevents SQL injection)
        insert_query = """
            INSERT INTO hazards (
                latitude, longitude, class_id, class_name, severity, 
                confidence, bbox, image_path, timestamp, status, 
                report_count, last_seen, notes
            ) VALUES (
                :latitude, :longitude, :class_id, :class_name, :severity,
                :confidence, :bbox, :image_path, :timestamp, :status,
                :report_count, :last_seen, :notes
            )
        """
        
        cursor.executemany(insert_query, hazards)
        
        conn.commit()
        
        print(f"✅ Successfully inserted {len(hazards)} sample hazards!\n")
        
        # Display summary statistics
        print("📊 Sample Data Summary:")
        
        # Count by city (approximate)
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN latitude BETWEEN 28.5 AND 28.7 THEN 'Delhi'
                    WHEN latitude BETWEEN 18.9 AND 19.2 THEN 'Mumbai'
                    WHEN latitude BETWEEN 12.9 AND 13.1 THEN 'Bangalore'
                    WHEN latitude BETWEEN 22.5 AND 22.7 THEN 'Kolkata'
                    ELSE 'Other'
                END as city,
                COUNT(*) as count
            FROM hazards
            GROUP BY city
        """)
        
        for city, count in cursor.fetchall():
            print(f"  {city}: {count} hazards")
        
        print()
        
        # Count by severity
        cursor.execute("""
            SELECT severity, COUNT(*) as count
            FROM hazards
            GROUP BY severity
            ORDER BY 
                CASE severity
                    WHEN 'critical' THEN 1
                    WHEN 'high' THEN 2
                    WHEN 'medium' THEN 3
                    WHEN 'low' THEN 4
                END
        """)
        
        print("  By Severity:")
        for severity, count in cursor.fetchall():
            print(f"    {severity.capitalize()}: {count}")
        
        print()
        
        # Count by class
        cursor.execute("""
            SELECT class_name, COUNT(*) as count
            FROM hazards
            GROUP BY class_name
            ORDER BY count DESC
        """)
        
        print("  By Type:")
        for class_name, count in cursor.fetchall():
            print(f"    {class_name.replace('_', ' ').title()}: {count}")
        
        print()
        
    except sqlite3.Error as e:
        print(f"❌ Error inserting sample data: {e}")
        raise


# ============================================================================
# Database Utilities
# ============================================================================

def verify_database(conn: sqlite3.Connection) -> bool:
    """
    Verify database integrity and structure.
    
    Args:
        conn: SQLite database connection
        
    Returns:
        True if database is valid, False otherwise
    """
    cursor = conn.cursor()
    
    try:
        print("🔍 Verifying database integrity...")
        
        # Check if hazards table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='hazards'
        """)
        
        if not cursor.fetchone():
            print("  ❌ Hazards table not found")
            return False
        
        print("  ✅ Hazards table exists")
        
        # Check table structure
        cursor.execute("PRAGMA table_info(hazards)")
        columns = {row[1] for row in cursor.fetchall()}
        
        required_columns = {
            'id', 'latitude', 'longitude', 'class_id', 'class_name',
            'severity', 'confidence', 'timestamp', 'status'
        }
        
        if not required_columns.issubset(columns):
            missing = required_columns - columns
            print(f"  ❌ Missing required columns: {missing}")
            return False
        
        print("  ✅ All required columns present")
        
        # Check indexes
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND tbl_name='hazards'
        """)
        
        indexes = {row[0] for row in cursor.fetchall()}
        expected_indexes = {
            'idx_hazards_location',
            'idx_hazards_timestamp',
            'idx_hazards_status'
        }
        
        if expected_indexes.issubset(indexes):
            print("  ✅ All indexes created")
        else:
            print("  ⚠️ Some indexes missing (will create)")
        
        # Count records
        cursor.execute("SELECT COUNT(*) FROM hazards")
        count = cursor.fetchone()[0]
        print(f"  📊 Total records: {count}")
        
        print("✅ Database verification complete!\n")
        return True
        
    except sqlite3.Error as e:
        print(f"❌ Error verifying database: {e}")
        return False


def clear_database(conn: sqlite3.Connection) -> None:
    """
    Clear all data from the database (keep schema).
    
    Args:
        conn: SQLite database connection
    """
    cursor = conn.cursor()
    
    try:
        print("🗑️  Clearing existing data...")
        cursor.execute("DELETE FROM hazards")
        conn.commit()
        print("✅ Database cleared!\n")
        
    except sqlite3.Error as e:
        print(f"❌ Error clearing database: {e}")
        raise


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main database initialization function."""
    
    print("=" * 60)
    print("  RoadGuard Database Initialization")
    print("=" * 60)
    print()
    
    try:
        # Ensure data directory exists
        data_dir = DB_PATH.parent
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Data directory: {data_dir}")
        print()
        
        # Check if database already exists
        db_exists = DB_PATH.exists()
        
        if db_exists:
            print(f"⚠️  Database already exists at: {DB_PATH}")
            response = input("Do you want to recreate it? (yes/no): ").strip().lower()
            
            if response not in ['yes', 'y']:
                print("❌ Initialization cancelled.")
                return
            
            # Backup existing database
            backup_path = DB_PATH.with_suffix('.backup.db')
            import shutil
            shutil.copy2(DB_PATH, backup_path)
            print(f"💾 Backup created at: {backup_path}")
            print()
        
        # Connect to database
        print(f"🔌 Connecting to database: {DB_PATH}")
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
        print("✅ Connected successfully!\n")
        
        # Clear existing data if recreating
        if db_exists:
            clear_database(conn)
        
        # Create tables
        create_tables(conn)
        
        # Ask if user wants sample data
        print("Do you want to populate the database with sample hazards?")
        response = input("(Recommended for testing) (yes/no): ").strip().lower()
        print()
        
        if response in ['yes', 'y']:
            insert_sample_hazards(conn)
        else:
            print("ℹ️  Skipping sample data insertion.")
            print("   You can add hazards through the application.\n")
        
        # Verify database
        verify_database(conn)
        
        # Close connection
        conn.close()
        print("🔌 Database connection closed.")
        print()
        
        print("=" * 60)
        print("  ✅ Database initialization complete!")
        print("=" * 60)
        print()
        print(f"Database location: {DB_PATH}")
        print("You can now run the Streamlit application:")
        print("  streamlit run app/main.py")
        print()
        
    except sqlite3.Error as e:
        print(f"\n❌ Database error: {e}")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n❌ Initialization interrupted by user.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# Command Line Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Initialize RoadGuard SQLite database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/init_database.py                    # Interactive mode
  python scripts/init_database.py --with-samples     # Create with sample data
  python scripts/init_database.py --force            # Recreate without prompt
        """
    )
    
    parser.add_argument(
        '--with-samples',
        action='store_true',
        help='Automatically populate with sample data'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Recreate database without prompting'
    )
    
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing database'
    )
    
    args = parser.parse_args()
    
    # Handle verify-only mode
    if args.verify_only:
        if not DB_PATH.exists():
            print(f"❌ Database not found at: {DB_PATH}")
            sys.exit(1)
        
        conn = sqlite3.connect(str(DB_PATH))
        verify_database(conn)
        conn.close()
        sys.exit(0)
    
    # Override interactive prompts if flags provided
    if args.force:
        # Monkey-patch input to always return 'yes'
        __builtins__.input = lambda _: 'yes'
    
    if args.with_samples:
        # Store original input
        original_input = input
        # Override only the sample data prompt
        call_count = [0]
        def custom_input(prompt):
            call_count[0] += 1
            if 'sample' in prompt.lower():
                return 'yes'
            return original_input(prompt)
        __builtins__.input = custom_input
    
    main()

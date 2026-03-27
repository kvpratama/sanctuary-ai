supabase gen types --lang=python --local > src/db/database_types.py
supabase db dump --local > src/db/schema.sql

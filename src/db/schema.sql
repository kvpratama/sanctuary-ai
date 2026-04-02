


SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;


CREATE EXTENSION IF NOT EXISTS "pg_net" WITH SCHEMA "extensions";






COMMENT ON SCHEMA "public" IS 'standard public schema';



CREATE EXTENSION IF NOT EXISTS "pg_graphql" WITH SCHEMA "graphql";






CREATE EXTENSION IF NOT EXISTS "pg_stat_statements" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "pgcrypto" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "supabase_vault" WITH SCHEMA "vault";






CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "vector" WITH SCHEMA "extensions";






CREATE OR REPLACE FUNCTION "public"."get_sorted_documents"("search_query" "text" DEFAULT NULL::"text", "limit_count" integer DEFAULT NULL::integer, "offset_count" integer DEFAULT 0) RETURNS TABLE("id" "uuid", "name" "text", "author" "text", "thumbnail_url" "text", "upload_date" timestamp with time zone, "last_accessed" timestamp with time zone, "page_count" integer, "current_page" integer, "total_count" bigint)
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
declare
  user_uuid uuid;
begin
  user_uuid := auth.uid();
  
  if user_uuid is null then
    raise exception 'Not authenticated';
  end if;

  return query
  with filtered_docs as (
    select 
      d.id,
      d.name,
      d.author,
      d.thumbnail_url,
      d.upload_date,
      d.last_accessed,
      d.page_count,
      d.current_page,
      case 
        when d.last_accessed is null and d.upload_date >= now() - interval '7 days' then 0
        when d.last_accessed is not null then 1
        else 2
      end as sort_bucket
    from public.documents d
    where d.user_id = user_uuid
      and (
        search_query is null 
        or d.name ilike '%' || search_query || '%'
        or d.author ilike '%' || search_query || '%'
      )
  ),
  total as (
    select count(*) as cnt from filtered_docs
  )
  select 
    f.id,
    f.name,
    f.author,
    f.thumbnail_url,
    f.upload_date,
    f.last_accessed,
    f.page_count,
    f.current_page,
    t.cnt as total_count
  from filtered_docs f
  cross join total t
  order by f.sort_bucket, coalesce(f.last_accessed, f.upload_date) desc
  limit limit_count
  offset offset_count;
end;
$$;


ALTER FUNCTION "public"."get_sorted_documents"("search_query" "text", "limit_count" integer, "offset_count" integer) OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."match_document_embeddings"("query_embedding" "extensions"."vector", "filter" "jsonb" DEFAULT '{}'::"jsonb", "match_count" integer DEFAULT 5) RETURNS TABLE("id" "uuid", "content" "text", "metadata" "jsonb", "similarity" double precision)
    LANGUAGE "plpgsql"
    SET "search_path" TO 'extensions', 'public'
    AS $$
begin
  return query
  select
    de.id,
    de.content,
    de.metadata,
    1 - (de.embedding <=> query_embedding) as similarity
  from public.document_embeddings de
  where
        de.user_id     = (filter->>'user_id')::uuid
    and de.document_id = (filter->>'document_id')::uuid
  order by de.embedding <=> query_embedding
  limit match_count;
end;
$$;


ALTER FUNCTION "public"."match_document_embeddings"("query_embedding" "extensions"."vector", "filter" "jsonb", "match_count" integer) OWNER TO "postgres";

SET default_tablespace = '';

SET default_table_access_method = "heap";


CREATE TABLE IF NOT EXISTS "public"."document_embeddings" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "document_id" "uuid" NOT NULL,
    "chunk_key" "text" NOT NULL,
    "content" "text" NOT NULL,
    "embedding" "extensions"."vector"(768) NOT NULL,
    "metadata" "jsonb" DEFAULT '{}'::"jsonb" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."document_embeddings" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."documents" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "name" "text" NOT NULL,
    "size" bigint NOT NULL,
    "blob_url" "text" NOT NULL,
    "thumbnail_url" "text",
    "author" "text",
    "page_count" integer,
    "upload_date" timestamp with time zone DEFAULT "now"() NOT NULL,
    "last_accessed" timestamp with time zone,
    "current_page" integer DEFAULT 0 NOT NULL,
    "ingested_at" timestamp with time zone,
    "is_ingesting" boolean DEFAULT false NOT NULL,
    CONSTRAINT "documents_current_page_check" CHECK (("current_page" >= 0)),
    CONSTRAINT "documents_size_check" CHECK (("size" > 0))
);


ALTER TABLE "public"."documents" OWNER TO "postgres";


ALTER TABLE ONLY "public"."document_embeddings"
    ADD CONSTRAINT "document_embeddings_document_id_chunk_key_key" UNIQUE ("document_id", "chunk_key");



ALTER TABLE ONLY "public"."document_embeddings"
    ADD CONSTRAINT "document_embeddings_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."documents"
    ADD CONSTRAINT "documents_id_user_id_key" UNIQUE ("id", "user_id");



ALTER TABLE ONLY "public"."documents"
    ADD CONSTRAINT "documents_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."documents"
    ADD CONSTRAINT "documents_user_id_name_key" UNIQUE ("user_id", "name");



CREATE INDEX "document_embeddings_user_id_document_id_idx" ON "public"."document_embeddings" USING "btree" ("user_id", "document_id");



CREATE INDEX "idx_documents_last_accessed" ON "public"."documents" USING "btree" ("user_id", "last_accessed" DESC NULLS LAST);



CREATE INDEX "idx_documents_upload_date" ON "public"."documents" USING "btree" ("user_id", "upload_date" DESC);



ALTER TABLE ONLY "public"."documents"
    ADD CONSTRAINT "documents_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."document_embeddings"
    ADD CONSTRAINT "embeddings_document_owner_fkey" FOREIGN KEY ("document_id", "user_id") REFERENCES "public"."documents"("id", "user_id") ON DELETE CASCADE;



CREATE POLICY "Users can delete own documents" ON "public"."documents" FOR DELETE USING ((( SELECT "auth"."uid"() AS "uid") = "user_id"));



CREATE POLICY "Users can delete own embeddings" ON "public"."document_embeddings" FOR DELETE USING ((( SELECT "auth"."uid"() AS "uid") = "user_id"));



CREATE POLICY "Users can insert own documents" ON "public"."documents" FOR INSERT WITH CHECK ((( SELECT "auth"."uid"() AS "uid") = "user_id"));



CREATE POLICY "Users can insert own embeddings" ON "public"."document_embeddings" FOR INSERT WITH CHECK ((( SELECT "auth"."uid"() AS "uid") = "user_id"));



CREATE POLICY "Users can read own documents" ON "public"."documents" FOR SELECT USING ((( SELECT "auth"."uid"() AS "uid") = "user_id"));



CREATE POLICY "Users can read own embeddings" ON "public"."document_embeddings" FOR SELECT USING ((( SELECT "auth"."uid"() AS "uid") = "user_id"));



CREATE POLICY "Users can update own documents" ON "public"."documents" FOR UPDATE USING ((( SELECT "auth"."uid"() AS "uid") = "user_id"));



ALTER TABLE "public"."document_embeddings" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."documents" ENABLE ROW LEVEL SECURITY;




ALTER PUBLICATION "supabase_realtime" OWNER TO "postgres";





GRANT USAGE ON SCHEMA "public" TO "postgres";
GRANT USAGE ON SCHEMA "public" TO "anon";
GRANT USAGE ON SCHEMA "public" TO "authenticated";
GRANT USAGE ON SCHEMA "public" TO "service_role";





















































































































































































































































































































































































































































































































GRANT ALL ON FUNCTION "public"."get_sorted_documents"("search_query" "text", "limit_count" integer, "offset_count" integer) TO "anon";
GRANT ALL ON FUNCTION "public"."get_sorted_documents"("search_query" "text", "limit_count" integer, "offset_count" integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."get_sorted_documents"("search_query" "text", "limit_count" integer, "offset_count" integer) TO "service_role";

































GRANT ALL ON TABLE "public"."document_embeddings" TO "anon";
GRANT ALL ON TABLE "public"."document_embeddings" TO "authenticated";
GRANT ALL ON TABLE "public"."document_embeddings" TO "service_role";



GRANT ALL ON TABLE "public"."documents" TO "anon";
GRANT ALL ON TABLE "public"."documents" TO "authenticated";
GRANT ALL ON TABLE "public"."documents" TO "service_role";









ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "service_role";
































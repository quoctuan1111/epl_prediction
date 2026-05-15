-- Create `profiles` table for Supabase
-- Run this in Supabase SQL editor

create extension if not exists pgcrypto;

create table if not exists profiles (
  id uuid primary key,
  nickname text,
  email text,
  created_at timestamptz default now()
);

-- Optional: ensure id matches auth.users if you're using Supabase Auth
-- If using Supabase Auth, consider creating profiles with id uuid references auth.users(id).

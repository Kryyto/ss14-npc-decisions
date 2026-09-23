create table predictions (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  job text not null,
  lang text not null,
  message text not null,
  answers jsonb not null,
  latency_ms integer,
  error text
);

create index predictions_job_created_at_idx on predictions (job, created_at desc);

alter table predictions enable row level security;

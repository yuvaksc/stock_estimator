-- ════════════════════════════════════════════════════════════════════════════
--  Real-Time Stock-Sentiment AI System — database schema
--  Run this in the Supabase SQL Editor (or against any Postgres).
--  It is idempotent (safe to run more than once).
-- ════════════════════════════════════════════════════════════════════════════

-- stocks reference
create table if not exists stocks (
  ticker        text primary key,
  company_name  text,
  created_at    timestamptz default now()
);

-- live price ticks
create table if not exists stock_ticks (
  id      bigint generated always as identity primary key,
  ticker  text references stocks(ticker) on delete cascade,
  price   numeric not null,
  volume  bigint,
  ts      timestamptz not null,
  source  text default 'finnhub'
);
create index if not exists idx_stock_ticks_ticker_ts on stock_ticks (ticker, ts desc);

-- social posts (Reddit / other social text) — historically "tweets"
create table if not exists posts (
  id           bigint generated always as identity primary key,
  ticker       text references stocks(ticker) on delete cascade,
  platform     text default 'reddit',
  external_id  text unique,
  author       text,
  body         text not null,
  created_at   timestamptz,
  ingested_at  timestamptz default now()
);
create index if not exists idx_posts_ticker on posts (ticker, ingested_at desc);

-- sentiment signals (one row per scored post; ensemble of Groq + FinBERT)
create table if not exists sentiment_signals (
  id                    bigint generated always as identity primary key,
  post_id               bigint references posts(id) on delete cascade,
  ticker                text references stocks(ticker) on delete cascade,
  sentiment             text check (sentiment in ('bullish','bearish','neutral')),
  confidence            numeric check (confidence between 0 and 1),
  impact_horizon_hours  int,
  model                 text,        -- e.g. 'ensemble:groq+finbert'
  finbert_score         numeric,     -- P(pos) - P(neg)
  rationale             text,
  created_at            timestamptz default now()
);
create index if not exists idx_sentiment_ticker_created on sentiment_signals (ticker, created_at desc);

-- price predictions (output of the multimodal regressor served behind /predict)
create table if not exists predictions (
  id               bigint generated always as identity primary key,
  ticker           text references stocks(ticker) on delete cascade,
  predicted_price  numeric,
  horizon          text,            -- e.g. '1d'
  model_version    text,
  features_hash    text,
  created_at       timestamptz default now()
);
create index if not exists idx_predictions_ticker_created on predictions (ticker, created_at desc);

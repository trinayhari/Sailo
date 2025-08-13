# Sailo - Supabase PostgreSQL Integration

A React + TypeScript + Vite application with complete Supabase PostgreSQL database integration.

## Features

- **Complete Supabase Setup** - Ready-to-use Supabase client configuration
- **Custom React Hooks** - Powerful hooks for database operations (CRUD)
- **Data Service Layer** - Utility functions for direct database access
- **TypeScript Support** - Fully typed database operations
- **Error Handling** - Comprehensive error management
- **Real-time Updates** - Built-in data refetching capabilities
- **Example Components** - Interactive demo components

## Quick Start

### 1. Environment Setup

Copy the environment template and add your Supabase credentials:

```bash
cp .env.example .env.local
```

Edit `.env.local` with your Supabase project details:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default tseslint.config([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

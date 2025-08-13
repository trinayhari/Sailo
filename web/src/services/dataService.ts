import { supabase } from '../lib/supabase'
import type { PostgrestError } from '@supabase/supabase-js'

// Generic data service for common database operations
export class DataService {
  // Fetch all records from a table
  static async fetchAll<T>(tableName: string, select = '*'): Promise<{
    data: T[] | null
    error: PostgrestError | null
  }> {
    const { data, error } = await supabase
      .from(tableName)
      .select(select)

    return { data: data as T[] | null, error }
  }

  // Fetch a single record by ID
  static async fetchById<T>(
    tableName: string,
    id: string | number,
    idColumn = 'id',
    select = '*'
  ): Promise<{
    data: T | null
    error: PostgrestError | null
  }> {
    const { data, error } = await supabase
      .from(tableName)
      .select(select)
      .eq(idColumn, id)
      .single()

    return { data: data as T | null, error }
  }

  // Fetch records with custom filters
  static async fetchWithFilters<T>(
    tableName: string,
    filters: Array<{
      column: string
      operator: 'eq' | 'neq' | 'gt' | 'gte' | 'lt' | 'lte' | 'like' | 'ilike' | 'in'
      value: any
    }>,
    options?: {
      select?: string
      orderBy?: { column: string; ascending?: boolean }
      limit?: number
      offset?: number
    }
  ): Promise<{
    data: T[] | null
    error: PostgrestError | null
  }> {
    let query = supabase
      .from(tableName)
      .select(options?.select || '*')

    // Apply filters
    filters.forEach(({ column, operator, value }) => {
      switch (operator) {
        case 'eq':
          query = query.eq(column, value)
          break
        case 'neq':
          query = query.neq(column, value)
          break
        case 'gt':
          query = query.gt(column, value)
          break
        case 'gte':
          query = query.gte(column, value)
          break
        case 'lt':
          query = query.lt(column, value)
          break
        case 'lte':
          query = query.lte(column, value)
          break
        case 'like':
          query = query.like(column, value)
          break
        case 'ilike':
          query = query.ilike(column, value)
          break
        case 'in':
          query = query.in(column, value)
          break
      }
    })

    // Apply ordering
    if (options?.orderBy) {
      query = query.order(options.orderBy.column, {
        ascending: options.orderBy.ascending ?? true
      })
    }

    // Apply pagination
    if (options?.limit) {
      query = query.limit(options.limit)
    }
    if (options?.offset) {
      query = query.range(options.offset, options.offset + (options.limit || 10) - 1)
    }

    const { data, error } = await query

    return { data: data as T[] | null, error }
  }

  // Insert a new record
  static async insert<T>(
    tableName: string,
    data: Partial<T> | Partial<T>[]
  ): Promise<{
    data: T[] | null
    error: PostgrestError | null
  }> {
    const { data: result, error } = await supabase
      .from(tableName)
      .insert(data)
      .select()

    return { data: result, error }
  }

  // Update records
  static async update<T>(
    tableName: string,
    updates: Partial<T>,
    filter: { column: string; value: any }
  ): Promise<{
    data: T[] | null
    error: PostgrestError | null
  }> {
    const { data, error } = await supabase
      .from(tableName)
      .update(updates)
      .eq(filter.column, filter.value)
      .select()

    return { data, error }
  }

  // Delete records
  static async delete(
    tableName: string,
    filter: { column: string; value: any }
  ): Promise<{
    error: PostgrestError | null
  }> {
    const { error } = await supabase
      .from(tableName)
      .delete()
      .eq(filter.column, filter.value)

    return { error }
  }

  // Execute a custom query with SQL
  static async executeQuery<T>(query: string): Promise<{
    data: T[] | null
    error: PostgrestError | null
  }> {
    const { data, error } = await supabase.rpc('execute_sql', { query })
    return { data, error }
  }

  // Count records in a table
  static async count(
    tableName: string,
    filters?: Array<{
      column: string
      operator: 'eq' | 'neq' | 'gt' | 'gte' | 'lt' | 'lte' | 'like' | 'ilike' | 'in'
      value: any
    }>
  ): Promise<{
    count: number | null
    error: PostgrestError | null
  }> {
    let query = supabase
      .from(tableName)
      .select('*', { count: 'exact', head: true })

    // Apply filters if provided
    if (filters) {
      filters.forEach(({ column, operator, value }) => {
        switch (operator) {
          case 'eq':
            query = query.eq(column, value)
            break
          case 'neq':
            query = query.neq(column, value)
            break
          case 'gt':
            query = query.gt(column, value)
            break
          case 'gte':
            query = query.gte(column, value)
            break
          case 'lt':
            query = query.lt(column, value)
            break
          case 'lte':
            query = query.lte(column, value)
            break
          case 'like':
            query = query.like(column, value)
            break
          case 'ilike':
            query = query.ilike(column, value)
            break
          case 'in':
            query = query.in(column, value)
            break
        }
      })
    }

    const { count, error } = await query

    return { count, error }
  }
}

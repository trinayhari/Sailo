import { useState, useEffect } from 'react'
import { supabase } from '../lib/supabase'
import type { PostgrestError } from '@supabase/supabase-js'

// Generic hook for fetching data from any table
export function useSupabaseQuery<T>(
  tableName: string,
  query?: {
    select?: string
    filter?: { column: string; operator: string; value: any }[]
    orderBy?: { column: string; ascending?: boolean }
    limit?: number
  }
) {
  const [data, setData] = useState<T[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<PostgrestError | null>(null)

  const fetchData = async () => {
    try {
      setLoading(true)
      setError(null)

      let queryBuilder = supabase
        .from(tableName)
        .select(query?.select || '*')

      // Apply filters
      if (query?.filter) {
        query.filter.forEach(({ column, operator, value }) => {
          switch (operator) {
            case 'eq':
              queryBuilder = queryBuilder.eq(column, value)
              break
            case 'neq':
              queryBuilder = queryBuilder.neq(column, value)
              break
            case 'gt':
              queryBuilder = queryBuilder.gt(column, value)
              break
            case 'gte':
              queryBuilder = queryBuilder.gte(column, value)
              break
            case 'lt':
              queryBuilder = queryBuilder.lt(column, value)
              break
            case 'lte':
              queryBuilder = queryBuilder.lte(column, value)
              break
            case 'like':
              queryBuilder = queryBuilder.like(column, value)
              break
            case 'ilike':
              queryBuilder = queryBuilder.ilike(column, value)
              break
            case 'in':
              queryBuilder = queryBuilder.in(column, value)
              break
            default:
              console.warn(`Unsupported operator: ${operator}`)
          }
        })
      }

      // Apply ordering
      if (query?.orderBy) {
        queryBuilder = queryBuilder.order(query.orderBy.column, {
          ascending: query.orderBy.ascending ?? true
        })
      }

      // Apply limit
      if (query?.limit) {
        queryBuilder = queryBuilder.limit(query.limit)
      }

      const { data: result, error: queryError } = await queryBuilder

      if (queryError) {
        setError(queryError)
      } else {
        setData((result as T[]) || [])
      }
    } catch (err) {
      setError(err as PostgrestError)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [tableName, JSON.stringify(query)])

  const refetch = () => {
    fetchData()
  }

  return { data, loading, error, refetch }
}

// Hook for inserting data
export function useSupabaseInsert<T>(tableName: string) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<PostgrestError | null>(null)

  const insert = async (data: Partial<T> | Partial<T>[]) => {
    try {
      setLoading(true)
      setError(null)

      const { data: result, error: insertError } = await supabase
        .from(tableName)
        .insert(data)
        .select()

      if (insertError) {
        setError(insertError)
        return null
      }

      return result
    } catch (err) {
      setError(err as PostgrestError)
      return null
    } finally {
      setLoading(false)
    }
  }

  return { insert, loading, error }
}

// Hook for updating data
export function useSupabaseUpdate<T>(tableName: string) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<PostgrestError | null>(null)

  const update = async (
    data: Partial<T>,
    filter: { column: string; value: any }
  ) => {
    try {
      setLoading(true)
      setError(null)

      const { data: result, error: updateError } = await supabase
        .from(tableName)
        .update(data)
        .eq(filter.column, filter.value)
        .select()

      if (updateError) {
        setError(updateError)
        return null
      }

      return result
    } catch (err) {
      setError(err as PostgrestError)
      return null
    } finally {
      setLoading(false)
    }
  }

  return { update, loading, error }
}

// Hook for deleting data
export function useSupabaseDelete(tableName: string) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<PostgrestError | null>(null)

  const deleteRecord = async (filter: { column: string; value: any }) => {
    try {
      setLoading(true)
      setError(null)

      const { error: deleteError } = await supabase
        .from(tableName)
        .delete()
        .eq(filter.column, filter.value)

      if (deleteError) {
        setError(deleteError)
        return false
      }

      return true
    } catch (err) {
      setError(err as PostgrestError)
      return false
    } finally {
      setLoading(false)
    }
  }

  return { deleteRecord, loading, error }
}

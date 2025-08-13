import { useState } from 'react'
import { useSupabaseQuery, useSupabaseInsert, useSupabaseUpdate, useSupabaseDelete } from '../hooks/useSupabaseData'
import { DataService } from '../services/dataService'

// Example component demonstrating data retrieval from PostgreSQL via Supabase
export function DataExample() {
  const [tableName, setTableName] = useState('options_trades') // Options trading table
  const [newRecord, setNewRecord] = useState({ symbol: '', contract_type: 'CALL', strike_price: '', premium: '' })
  const [updateId, setUpdateId] = useState('')
  const [updateData, setUpdateData] = useState({ symbol: '', premium: '' })

  // Using the custom hook to fetch data
  const { data, loading, error, refetch } = useSupabaseQuery(tableName, {
    orderBy: { column: 'created_at', ascending: false },
    limit: 10
  })

  // Using hooks for CRUD operations
  const { insert, loading: insertLoading, error: insertError } = useSupabaseInsert(tableName)
  const { update, loading: updateLoading, error: updateError } = useSupabaseUpdate(tableName)
  const { deleteRecord, loading: deleteLoading, error: deleteError } = useSupabaseDelete(tableName)

  const handleInsert = async () => {
    const result = await insert(newRecord)
    if (result) {
      setNewRecord({ symbol: '', contract_type: 'CALL', strike_price: '', premium: '' })
      refetch() // Refresh the data
    }
  }

  const handleUpdate = async () => {
    if (updateId) {
      const result = await update(updateData, { column: 'id', value: updateId })
      if (result) {
        setUpdateId('')
        setUpdateData({ symbol: '', premium: '' })
        refetch()
      }
    }
  }

  const handleDelete = async (id: string) => {
    const success = await deleteRecord({ column: 'id', value: id })
    if (success) {
      refetch()
    }
  }

  // Example of using DataService directly
  const handleFetchWithService = async () => {
    const { data: serviceData, error: serviceError } = await DataService.fetchAll(tableName)
    console.log('Data from service:', serviceData)
    if (serviceError) console.error('Service error:', serviceError)
  }

  if (loading) return <div className="loading">Loading data...</div>

  return (
    <div className="data-example">
      <h2>PostgreSQL Data via Supabase</h2>
      
      {/* Table Name Input */}
      <div className="table-input">
        <label>
          Table Name:
          <input
            type="text"
            value={tableName}
            onChange={(e) => setTableName(e.target.value)}
            placeholder="Enter table name"
          />
        </label>
        <button onClick={refetch}>Refresh Data</button>
        <button onClick={handleFetchWithService}>Fetch with Service</button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="error">
          <h3>Error:</h3>
          <p>{error.message}</p>
          <details>
            <summary>Error Details</summary>
            <pre>{JSON.stringify(error, null, 2)}</pre>
          </details>
        </div>
      )}

      {/* Insert New Record */}
      <div className="insert-section">
        <h3>Insert New Record</h3>
        <input
          type="text"
          placeholder="Symbol (e.g., AAPL)"
          value={newRecord.symbol}
          onChange={(e) => setNewRecord({ ...newRecord, symbol: e.target.value })}
        />
        <select
          value={newRecord.contract_type}
          onChange={(e) => setNewRecord({ ...newRecord, contract_type: e.target.value })}
        >
          <option value="CALL">CALL</option>
          <option value="PUT">PUT</option>
        </select>
        <input
          type="number"
          placeholder="Strike Price"
          value={newRecord.strike_price}
          onChange={(e) => setNewRecord({ ...newRecord, strike_price: e.target.value })}
        />
        <input
          type="number"
          step="0.01"
          placeholder="Premium"
          value={newRecord.premium}
          onChange={(e) => setNewRecord({ ...newRecord, premium: e.target.value })}
        />
        <button onClick={handleInsert} disabled={insertLoading}>
          {insertLoading ? 'Inserting...' : 'Insert'}
        </button>
        {insertError && <p className="error">Insert Error: {insertError.message}</p>}
      </div>

      {/* Update Record */}
      <div className="update-section">
        <h3>Update Record</h3>
        <input
          type="text"
          placeholder="Record ID"
          value={updateId}
          onChange={(e) => setUpdateId(e.target.value)}
        />
        <input
          type="text"
          placeholder="New Symbol"
          value={updateData.symbol}
          onChange={(e) => setUpdateData({ ...updateData, symbol: e.target.value })}
        />
        <input
          type="number"
          step="0.01"
          placeholder="New Premium"
          value={updateData.premium}
          onChange={(e) => setUpdateData({ ...updateData, premium: e.target.value })}
        />
        <button onClick={handleUpdate} disabled={updateLoading}>
          {updateLoading ? 'Updating...' : 'Update'}
        </button>
        {updateError && <p className="error">Update Error: {updateError.message}</p>}
      </div>

      {/* Data Display */}
      <div className="data-display">
        <h3>Data from {tableName} table:</h3>
        {data && data.length > 0 ? (
          <div className="data-grid">
            {data.map((record: any, index) => (
              <div key={record.id || index} className="data-record">
                <pre>{JSON.stringify(record, null, 2)}</pre>
                <button 
                  onClick={() => handleDelete(record.id)} 
                  disabled={deleteLoading}
                  className="delete-btn"
                >
                  Delete
                </button>
              </div>
            ))}
          </div>
        ) : (
          <p>No data found in table "{tableName}"</p>
        )}
        {deleteError && <p className="error">Delete Error: {deleteError.message}</p>}
      </div>

      {/* Advanced Query Example */}
      <div className="advanced-query">
        <h3>Advanced Query Examples</h3>
        <p>You can use the hooks with advanced filters:</p>
        <pre>{`
// Filter by email domain
const { data } = useSupabaseQuery('users', {
  filter: [{ column: 'email', operator: 'ilike', value: '%@gmail.com' }],
  orderBy: { column: 'created_at', ascending: false },
  limit: 5
})

// Filter by date range
const { data } = useSupabaseQuery('orders', {
  filter: [
    { column: 'created_at', operator: 'gte', value: '2024-01-01' },
    { column: 'status', operator: 'eq', value: 'completed' }
  ]
})
        `}</pre>
      </div>
    </div>
  )
}

export default DataExample

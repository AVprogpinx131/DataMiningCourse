import React, { useState } from 'react';
import { ArrowUpDown, ArrowLeftRight } from 'lucide-react';

const originalData = [
  [0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0], // S1
  [0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,0], // S2
  [0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0], // S3
  [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,1], // S4
  [0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,0], // S5
  [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,1], // S6
  [0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0], // S7
  [0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0], // S8
  [0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,0]  // S9
];

const SupplierProductMatrix = () => {
  const [matrix, setMatrix] = useState(originalData);
  const [supplierOrder, setSupplierOrder] = useState([...Array(9)].map((_, i) => i));
  const [productOrder, setProductOrder] = useState([...Array(16)].map((_, i) => i));
  
  const calculateSimilarity = (arr1, arr2) => {
    return arr1.reduce((acc, val, idx) => acc + (val === arr2[idx] ? 1 : 0), 0);
  };

  const getColumnData = (colIdx) => {
    return matrix.map(row => row[colIdx]);
  };

  const reorderSuppliers = () => {
    const newOrder = [];
    const used = new Set();
    let current = 0;
    newOrder.push(current);
    used.add(current);

    while (newOrder.length < supplierOrder.length) {
      let bestSimilarity = -1;
      let bestSupplier = -1;
      const currentRow = matrix[newOrder[newOrder.length - 1]];

      for (let i = 0; i < matrix.length; i++) {
        if (!used.has(i)) {
          const similarity = calculateSimilarity(currentRow, matrix[i]);
          if (similarity > bestSimilarity) {
            bestSimilarity = similarity;
            bestSupplier = i;
          }
        }
      }

      newOrder.push(bestSupplier);
      used.add(bestSupplier);
    }

    setSupplierOrder(newOrder);
  };

  const reorderProducts = () => {
    const newOrder = [];
    const used = new Set();
    let current = 0;
    newOrder.push(current);
    used.add(current);

    while (newOrder.length < productOrder.length) {
      let bestSimilarity = -1;
      let bestProduct = -1;
      const currentCol = getColumnData(newOrder[newOrder.length - 1]);

      for (let i = 0; i < matrix[0].length; i++) {
        if (!used.has(i)) {
          const similarity = calculateSimilarity(currentCol, getColumnData(i));
          if (similarity > bestSimilarity) {
            bestSimilarity = similarity;
            bestProduct = i;
          }
        }
      }

      newOrder.push(bestProduct);
      used.add(bestProduct);
    }

    setProductOrder(newOrder);
  };

  const reorderBoth = () => {
    reorderSuppliers();
    reorderProducts();
  };

  return (
    <div className="p-4">
      <div className="mb-4 space-x-4">
        <button 
          onClick={reorderSuppliers}
          className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          <ArrowUpDown size={16} />
          Reorder Suppliers
        </button>
        
        <button 
          onClick={reorderProducts}
          className="flex items-center gap-2 px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
        >
          <ArrowLeftRight size={16} />
          Reorder Products
        </button>
        
        <button 
          onClick={reorderBoth}
          className="flex items-center gap-2 px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600"
        >
          <ArrowUpDown size={16} />
          Reorder Both
        </button>
      </div>
      
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse">
          <thead>
            <tr>
              <th className="border p-2">Supplier</th>
              {productOrder.map(prodIdx => (
                <th key={prodIdx} className="border p-2">P{prodIdx + 1}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {supplierOrder.map((supplierIdx, i) => (
              <tr key={i}>
                <td className="border p-2 font-medium">S{supplierIdx + 1}</td>
                {productOrder.map((prodIdx, j) => (
                  <td 
                    key={j} 
                    className={`border p-2 text-center ${
                      matrix[supplierIdx][prodIdx] === 1 ? 'bg-blue-500 text-white' : 'bg-gray-50'
                    }`}
                  >
                    {matrix[supplierIdx][prodIdx]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default SupplierProductMatrix;
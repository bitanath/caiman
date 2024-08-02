import type { NextApiRequest, NextApiResponse } from 'next'
 
export default function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === 'POST') {
    // Process a POST request
  } else {
    // Handle any other HTTP method
  }
}

export const config = {
    api: {
      bodyParser: {
        sizeLimit: '1mb',
      },
    },
    maxDuration: 5,
}
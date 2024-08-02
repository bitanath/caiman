import { clientConfig, adminConfig } from '@/config'
import { initializeApp } from 'firebase/app'
import * as admin from 'firebase-admin'

//client side firestore queries are secured using the user that requested them
export const app = initializeApp(clientConfig)
//server side firestore queries are secured using the admin user
export const serverApp = admin.initializeApp({ credential: admin.credential.cert(adminConfig as admin.ServiceAccount) },`${(Math.random()*100).toFixed(5).toString()}`)
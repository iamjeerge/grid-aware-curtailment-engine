/**
 * API client for the Grid-Aware Curtailment Engine.
 */

import axios from 'axios';
import type {
  DemoScenario,
  HealthResponse,
  OptimizationListResponse,
  OptimizationRequest,
  OptimizationResult,
  SystemInfo,
} from './types';

const api = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

// =============================================================================
// Optimization Endpoints
// =============================================================================

export async function createOptimization(
  request: OptimizationRequest
): Promise<OptimizationResult> {
  const response = await api.post<OptimizationResult>('/optimizations/', request);
  return response.data;
}

export async function listOptimizations(
  page: number = 1,
  pageSize: number = 10
): Promise<OptimizationListResponse> {
  const response = await api.get<OptimizationListResponse>('/optimizations/', {
    params: { page, page_size: pageSize },
  });
  return response.data;
}

export async function getOptimization(id: string): Promise<OptimizationResult> {
  const response = await api.get<OptimizationResult>(`/optimizations/${id}`);
  return response.data;
}

export async function deleteOptimization(id: string): Promise<void> {
  await api.delete(`/optimizations/${id}`);
}

// =============================================================================
// Demo Endpoints
// =============================================================================

export async function listDemoScenarios(): Promise<DemoScenario[]> {
  const response = await api.get<DemoScenario[]>('/demos/scenarios');
  return response.data;
}

export async function runDemoScenario(scenarioId: string): Promise<OptimizationResult> {
  const response = await api.post<OptimizationResult>(`/demos/run/${scenarioId}`);
  return response.data;
}

// =============================================================================
// Health & System Endpoints
// =============================================================================

export async function getHealth(): Promise<HealthResponse> {
  const response = await axios.get<HealthResponse>('/health');
  return response.data;
}

export async function getSystemInfo(): Promise<SystemInfo> {
  const response = await axios.get<SystemInfo>('/system');
  return response.data;
}

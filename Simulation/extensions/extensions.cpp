#include <algorithm>
#include <iostream>
#include <limits>
#include <math.h>
#include <queue>
#include <vector>

using namespace std;

/* ========== SEARCH_GRASP_PLACE_GRAPH ========== */

#define CLEAN_SEARCH_GRASP_PLACE_GRAPH {delete[] cost; delete[] estTotalCost; delete[] visited;}

/* Defines an operator which returns true iff the left item has greater estimated total cost than
 * the right item. This is used as the "less than" operator in the priority queue, so that minimum
 * cost elements have highest priority.
 */
class CompareCost
{
private:

  int nPlaces;
  float* estTotalCost;

public:

  /* We need a reference to the total cost data structure when comparing plans.
   * - Input arg1: A table of estimated total cost for each node, same size as the grasp-place table.
   * - Input arg2: Number of places
   * */
  CompareCost(float* arg1, int arg2)
  {
    estTotalCost = arg1;
    nPlaces = arg2;
  }

  /* If true, r has higher priority than l.
   */
  bool operator()(const vector< pair<int, int> >& l, const vector< pair<int, int> >& r) const
  {
    pair<int, int> lPair = l.back();
    pair<int, int> rPair = r.back();
    return estTotalCost[nPlaces * lPair.first + lPair.second] >
           estTotalCost[nPlaces * rPair.first + rPair.second];
  }
};

/* Copies a plan from a vector of pairs to a flat array.
 * - Input item: List of pairs of the form: [(grasp, place)_1, ..., (grasp, place)_n].
 * - Output plan: Integer array of 2*maxPlanLength initialized to -1. Will have form:
 *   [grasp_1, place_1, ..., grasp_n, place_n, ..., -1, -1], where n <= maxPlanLength.
 */
void PlanVectorToPlanArray(vector< pair<int, int> > item, int* plan)
{
  for (int i = 0; i < item.size(); i++)
  {
    plan[2 * i + 0] = item[i].first;
    plan[2 * i + 1] = item[i].second;
  }
}

/* Returns true if the given place has been visited in the given plan; returns false otherwise.
 * - Input place: The place (column index into grasp-place table) to check for.
 * - Input plan: The plan to search in. List of pairs of the form:
 *   [(grasp, place)_1, ..., (grasp, place)_n].
 * - Input checkStart: Start checking the plan at this step. Should be in [0, plan.size() - 1].
 * - Input checkEnd: Stop checking the plan just before this step. Should be in [0, plan.size()].
 */
bool VisitedPlace(int place, vector< pair<int, int> > plan, int checkStart, int checkEnd)
{
  for (int i = checkStart; i < checkEnd; i++)
  {
    if (plan[i].second == place)
    {
      return true;
    }
  }
  return false;
}

/* Uses A* to search the grasp-place graph for a minimum-cost plan.
 * The heuristic is 0 for terminal nodes and stepCost + minGoalGraspCost + minGoalPlaceCost for
 * non-terminal nodes.
 * - Input nGrasps: Number of rows in the grasp-place table.
 * - Input nPlaces: Number of columns in the grasp-place table.
 * - Input graspPlaceTable: Flattened matrix of size nGrasps * nPlaces in C-order, i.e.,
 *   graspPlaceTable[nPlaces * i + j] indexes grasp i at place j. The values are -1 if the grasp is
 *   not reachable at the place, 0 if the grasp has not been checked for reachability at the place,
 *   and 1 if the grasp is reachable at the place. For this search, we are only interested in
 *   entries with a value of 1.
 * - Input placeTypes: An array of size nPlaces indicating the type of place. 0 indicates start, 1
 *   indicates temporary, and 2 indicates goal.
 * - Input graspCosts: The cost of each grasp. Array of size nGrasps.
 * - Input placeCosts: The cost of each place. Array of size nPlaces.
 * - Input stepCost: The cost for each step in the plan.
 * - Input maxSearchDepth: The maximum plan length.
 * - Output plan: An array with size 2*maxSearchDepth, initialized to negative values. Will be
 *   populated with the plan with form: [grasp_1, place_1, ..., grasp_n, place_n, ..., -1, -1],
 *   where n <= maxPlanLength.
 */
extern "C" float SearchGraspPlaceGraph(int nGrasps, int nPlaces, int* graspPlaceTable,
  int* placeTypes, float* graspCosts, float* placeCosts, float stepCost, float remainingTime,
  int maxSearchDepth, int* plan)
{
  /* --- Summary of input semantics:

  graspPlaceTable[nPlaces * grasp + place]

  graspPlaceTable:
    -1: invalid
     0: not checked
     1: valid

  placeTypes:
     0: start
     1: temporary
     2: goal

  -------------------------------- */

  // initialize timer
  time_t startTime = time(0);

  // find the minimum goal grasp cost and the minimum goal place cost
  float minGoalGraspCost = numeric_limits<float>::infinity();
  float minGoalPlaceCost = numeric_limits<float>::infinity();
  for (int i = 0; i < nPlaces; i++)
  {
    if (placeTypes[i] == 2)
    {
      for (int j = 0; j < nGrasps; j++)
      {
        if (graspPlaceTable[nPlaces * j + i] == 1)
        {
          if (minGoalGraspCost > graspCosts[j])
          {
            minGoalGraspCost = graspCosts[j];
          }
          if (minGoalPlaceCost > placeCosts[i])
          {
            minGoalPlaceCost = placeCosts[i];
          }
        }
      }
    }
  }

  // check if there are any reachable goals
  if (isinf(minGoalGraspCost))
  {
    return numeric_limits<float>::infinity();
  }

  // for each node, initialize cost, estimated total cost, and visited
  int tableSize = nGrasps * nPlaces;
  float* cost = new float[tableSize];
  float* estTotalCost = new float[tableSize];
  bool* visited = new bool[tableSize];
  fill_n(cost, tableSize, numeric_limits<float>::infinity());
  fill_n(estTotalCost, tableSize, numeric_limits<float>::infinity());
  fill_n(visited, tableSize, false);

  // initialize queue
  priority_queue< vector< pair<int, int> >, vector< vector< pair<int, int> > >, CompareCost>
    q(CompareCost(cost, nPlaces));

  for (int i = 0; i < nGrasps; i++)
  {
    if (graspPlaceTable[nPlaces * i + 0] > 0)
    {
      cost[nPlaces * i + 0] = stepCost + graspCosts[i] + placeCosts[0];
      estTotalCost[nPlaces * i + 0] = cost[nPlaces * i + 0] + stepCost + minGoalGraspCost +
        minGoalPlaceCost;
      visited[nPlaces * i + 0] = true;
      vector< pair<int, int> > item;
      pair<int, int> step(i, 0);
      item.push_back(step);
      q.push(item);
    }
  }

  // A*
  while (q.size() > 0)
  {
    vector< pair<int, int> > item = q.top();
    q.pop();

    int grasp = item.back().first;
    int place = item.back().second;
    float itemCost = cost[nPlaces * grasp + place];

    if (placeTypes[place] == 2)
    {
      // place pose is terminal
      PlanVectorToPlanArray(item, plan);
      CLEAN_SEARCH_GRASP_PLACE_GRAPH
      return itemCost;
    }

    if (item.size() == maxSearchDepth)
    {
      // do not expand nodes beyond maximum search depth
      continue;
    }

    // expand the current node by first adding the same grasp at different places
    for (int i = 1; i < nPlaces; i++)
    {
      if (i != place)
      {
        int newItemIdx = nPlaces * grasp + i;
        if (graspPlaceTable[newItemIdx] > 0)
        {
          // (there is no reason to visit the same temporary placement more than once)
          if (!(placeTypes[i] == 1 && VisitedPlace(i, item, 1, item.size())))
          {
            float tentativeNewItemCost = itemCost + stepCost + graspCosts[grasp] + placeCosts[i];
            if (tentativeNewItemCost < cost[newItemIdx])
            {
              cost[newItemIdx] = tentativeNewItemCost;
              estTotalCost[newItemIdx] = (placeTypes[i] == 2) ? tentativeNewItemCost :
                tentativeNewItemCost + stepCost + minGoalGraspCost + minGoalPlaceCost;

              if (!visited[newItemIdx])
              {
                visited[newItemIdx] = true;
                pair<int, int> step(grasp, i);
                vector< pair<int, int> > newItem(item);
                newItem.push_back(step);
                q.push(newItem);
              }
            } // end if(!placeTypes[i] -- 1 && ...

          }
        }
      }
    }

    // if this is a temporary placement, continue expanding by switching grasps
    if (placeTypes[place] == 1)
    {
      // (there is no reason to switch grasps at the same temporary place more than once)
      if (!VisitedPlace(place, item, 1, item.size() - 1))
      {
        for (int i = 0; i < nGrasps; i++)
        {
          if (i != grasp)
          {
            int newItemIdx = nPlaces * i + place;
            if (graspPlaceTable[newItemIdx] > 0)
            {
              float tentativeNewItemCost = itemCost + stepCost + graspCosts[i] + placeCosts[place];
              if (tentativeNewItemCost < cost[newItemIdx])
              {
                cost[newItemIdx] = tentativeNewItemCost;
                estTotalCost[newItemIdx] = tentativeNewItemCost + stepCost + minGoalGraspCost +
                  minGoalPlaceCost;
                if (!visited[newItemIdx])
                {
                  visited[newItemIdx] = true;
                  pair<int, int> step(i, place);
                  vector< pair<int, int> > newItem(item);
                  newItem.push_back(step);
                  q.push(newItem);
                }
              }
            }
          }
        }
      }
    }

  } // end while

  CLEAN_SEARCH_GRASP_PLACE_GRAPH
  return numeric_limits<float>::infinity();
}

/* ========== IS_OCCLUDED ========== */

/* Estimates if a point in a cloud is occluded by other points in the cloud from a point sensor.
 * Derives from dist_point_to_segment at http://geomalgorithms.com/a02-_lines.html.
 * - Input cloud: nx3 list of points for the occluding cloud, which also contains the target point.
 * - Input viewPoint: 3-element array, the view point to consider.
 * - Input queryPointIdx: The index into cloud indicating the target point of the query.
 * - Input nPoints: Number of points in the cloud, n.
 * - Input maxDistToRaySquared: Maximum squared Euclidean distance a point can be from the line
 *   segment between the view point and the query point in order for the line to be considered occluded.
 * - Returns isOccluded: Returns true if the query point is occluded from the input view point and 
 *   false otherwise.
 */
bool IsPointOccluded(float* cloud, float* viewPoint, int queryPointIdx, int nPoints, float maxDistToRaySquared)
{
  // Get the vector sensor -> queryPoint.
  float sensorToQueryPoint[3];
  sensorToQueryPoint[0] = cloud[3*queryPointIdx + 0] - viewPoint[0];
  sensorToQueryPoint[1] = cloud[3*queryPointIdx + 1] - viewPoint[1];
  sensorToQueryPoint[2] = cloud[3*queryPointIdx + 2] - viewPoint[2];
  
  float sensorToQueryPointLength = pow(sensorToQueryPoint[0], 2) +
                                   pow(sensorToQueryPoint[1], 2) + 
                                   pow(sensorToQueryPoint[2], 2);
  
  for (int i = 0; i < nPoints; i++)
  {
    // To avoid numerical precision issues, skip points near the query point.
    if (pow((cloud[3*queryPointIdx + 0] - cloud[3*i + 0]), 2) +
        pow((cloud[3*queryPointIdx + 1] - cloud[3*i + 1]), 2) + 
        pow((cloud[3*queryPointIdx + 2] - cloud[3*i + 2]), 2) <= maxDistToRaySquared)
    { continue; }
    
    // Get the vector sensor -> (candidate) occluder.
    float sensorToOccluder[3];  
    sensorToOccluder[0] = cloud[i*3 + 0] - viewPoint[0];
    sensorToOccluder[1] = cloud[i*3 + 1] - viewPoint[1];
    sensorToOccluder[2] = cloud[i*3 + 2] - viewPoint[2];
    
    // Project sensorToOccluder onto sensorToQuery point. This is the definition of the dot product.
    float projectionLength = sensorToQueryPoint[0] * sensorToOccluder[0] + 
                             sensorToQueryPoint[1] * sensorToOccluder[1] + 
                             sensorToQueryPoint[2] * sensorToOccluder[2];
                       
    if (projectionLength <= 0)
    {
      // The angle between sensorToOccluder and sensorToQuery exceeds 90 degrees. Thus, the candidate
      // occluder is behind the view point and cannot be an actual occluder.
      continue; 
    }
    
    if (sensorToQueryPointLength <= projectionLength)
    {
      // The candidate occluder must be beyond the query point, so it cannot be an actual occluder.
      continue;
    }
    
    // Compute the point that projects the occluder onto sensor -> query point. This will help us
    // determine how far the occluder is from the line segment sensor -> query point.
    float alpha = projectionLength / sensorToQueryPointLength;
    
    float closestPointOnLine[3];
    closestPointOnLine[0] = viewPoint[0] + alpha * sensorToQueryPoint[0];
    closestPointOnLine[1] = viewPoint[1] + alpha * sensorToQueryPoint[1];
    closestPointOnLine[2] = viewPoint[2] + alpha * sensorToQueryPoint[2];
    
    // If the occluder is less than maxDistToRay to the line segment sensor -> query point, then
    // we consider the ray/line segment to be occluded.
    float distanceToLine = pow((closestPointOnLine[0] - cloud[3*i + 0]), 2) + 
                           pow((closestPointOnLine[1] - cloud[3*i + 1]), 2) +
                           pow((closestPointOnLine[2] - cloud[3*i + 2]), 2);
    
    if (distanceToLine < maxDistToRaySquared)
    {
      // The candidate occluder is close to the line, so thus the ray is occluded.
      return true;
    }
  }
  
  // No points were found occluding the ray.
  return false;
}

/* Estimates which points are occluded from rays emitted from given view points.
 * - Input cloud: nx3 list of points: cloud = [x_1, y_1, z_1, x_2, y_2, z_2, ..., z_n].
 * - Input viewPoints: mx3 list of points. View points can be regarded as point sources that emit
 *   rays in all directions.
 * - Input nPoints: The number of points in the cloud, n.
 * - Input nViewPoints: The number of view points, m.
 * - Input maxDistToRay: If a point is between the view point and the target point and is within
 *   this Euclidean distance from the ray (i.e., the line segment between the view point and the
 *   target point), then the target point is considered to be occluded.
 * - Output occluded: mxn binary array, indicating which points are occluded from which view points.
 *   occluded = [o_view1_point1, ..., o_viewm_point1, o_view1_point2, ..., o_viewm_pointn].
 * - Returns 0 if successful, returns a negative value if there is an input error.
 */
extern "C" int IsOccluded(float* cloud, float* viewPoints, int nPoints, int nViewPoints,
  float maxDistToRay, bool* occluded)
{
  
  // input checking
  
  if (nPoints < 0) { return -1; }
  if (nViewPoints < 0) { return -2; }
  if (maxDistToRay < 0) { return -3; }
  
  // preprocessing
  
  float maxDistToRaySquared = pow(maxDistToRay, 2);
  
  // for each view point
  for (int i = 0; i < nViewPoints; i++)
  {
    // for each target point in the cloud
    for (int j = 0; j < nPoints; j++)
    {
      occluded[nViewPoints*j + i] = IsPointOccluded(
        cloud, viewPoints + 3*i, j, nPoints, maxDistToRaySquared);
    }
  }
  
  return 0;
}

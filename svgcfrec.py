import sys, getopt
import xml.etree.ElementTree as et
from re import sub
import numpy as np
from scipy.spatial.distance import pdist, squareform
import collections as col 
from itertools import combinations

inputfile = "VAT_10908_Vs.svg"
layerID = "Kopie"
outputfile = "processed-"+inputfile 
floatType = np.float32
pathAttributes = {"d":"", "fill":"none", "stroke": "blue", "stroke-width": "0.5"} 

allowedAnkle = np.pi/3


"""
Process commandline options
"""

try:
   opts, args = getopt.getopt(sys.argv[1:],"hl:f:o:", ["layerid=","floattype=","ofile="])
except getopt.GetoptError:
   print 'test.py [-l <layerid>] [-f <floattype>] [-o <outputfile>] inputfile'
   sys.exit(2)
for opt, arg in opts:
   if opt == '-h':
      print 'test.py [-l <layerid>] [-f <floattype>] [-o <outputfile>] inputfile' 
      sys.exit()
   elif opt in ("-l", "--layerid"):
      layerID = arg
   elif opt in ("-f", "--floattype"):
      floatType = np.dtype(arg)
   elif opt in ("-o", "--ofile"):
      outputfile = arg


"""
Read file and find layer with hand-drawn cuneiform writing
"""

tree = et.parse(inputfile)
root = tree.getroot()
layer = root.find("*[@id='"+layerID+"']")


"""
Determine curves which could belong to the cuneiform, enumerate them in 
the svg and calculate a reference point for each of those curves
"""

# calculate reference points for each curve
expectedSize = len(layer)
absPoints = np.empty([expectedSize,4,2],dtype=floatType)
refPoints = np.empty([expectedSize,2],dtype=floatType)
derivatives = np.empty([expectedSize,2],dtype=floatType)

def getAngle(v1,v2):
   # return angle between v1 and v2
   return np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

def getReferencePoint(points):
   # get intersection of lines defined by 
   # - point 0 and 1 and 
   # - point 2 and 3 respectively.
   # Note that points 1-3 are relative to point 0, so 
   #           (b - a) * r - ( d - c ) * s = c - a
   # maps to:     p1   * r + (p2 - p3) * s =  p2
   A = np.vstack((points[1], points[2]-points[3])).T
   x = np.linalg.solve(A,points[2].T)
   return points[0] + x[0] * points[1]
   

curvenr = 0
nonConformCurves = []
unit = np.array([1,0])
for g in layer:
   # get points
   path = g[0][0].get("d")
   formatted = sub("[a-zA-Z,]", " ", path.replace("-", " -"))
   pointsAsList = formatted.split()
   
   if len(pointsAsList) == 8: 
      # get absolute points
      points = np.array(pointsAsList, dtype=floatType).reshape((4,2))
      absPoints[curvenr][0] = points[0].T
      absPoints[curvenr][1:4] = (points[0] + points[1:4]).reshape(3,2)

      # get reference point and derivatives at end points for curve
      derivatives[curvenr][0] = getAngle(points[1],unit)      
      derivatives[curvenr][1] = getAngle(points[2]-points[3],unit)
      refPoints[curvenr] = getReferencePoint(points)
      
      # enumerate curve to find it again
      g.set("curvenr",str(curvenr))
      curvenr = curvenr + 1
   else:
      nonConformCurves.append(path)

# trim to real size
absPoints.resize((curvenr,4,2))
refPoints.resize((curvenr,2))


"""
Calculate the distances from the reference points to each other and
find groups of three curves belonging together
"""
def angleTooWide(alpha,beta,allowedAngle):
   return abs(abs(alpha-beta)-(np.pi/2)) < (np.pi/2)-allowedAngle


def meltPaths(absPoints, refPoints, derivatives, dist, idx):
   # return only index triples that appear three times 
   # (meaning: for each reference point in the tuple the same 
   # group of three including itself is found) 
   # note: cast to "tuple" makes the entries hashable as required
   closest = np.argsort(dist[idx,:][:,idx])
   count = col.Counter([tuple(x) for x in np.sort(closest[:,:3])])
   cuneiforms = [k for (k, v) in count.items() if v==3]
   
   maxDist = 0
   unUsedCurves = np.ones(len(idx))
   cfnr=0
   # calculate one path for each group of three paths
   for triple in cuneiforms:
      badTriple = False
      paths = absPoints[idx[triple,:],:,:]
      dx = derivatives[idx[triple,:],:]
      midpoint = np.mean(refPoints[idx[triple,:],:],axis=0)
            
      # determine which curves have to be turned around
      #d = squareform(pdist(dx.reshape(6,1)))
      d = squareform(pdist(paths[:,::3,:].reshape(6,2)))
      c = np.empty((2,2,2)) 
      c[0,0,0] = d[1,2]+d[3,4]+d[5,0]
      c[0,0,1] = d[1,2]+d[3,5]+d[4,0]
      c[0,1,0] = d[1,3]+d[2,4]+d[5,0]
      c[0,1,1] = d[1,3]+d[2,5]+d[4,0]
      c[1,0,0] = d[0,2]+d[3,4]+d[5,1]
      c[1,0,1] = d[0,2]+d[3,5]+d[4,1]
      c[1,1,0] = d[0,3]+d[2,4]+d[5,1]
      c[1,1,1] = d[0,3]+d[2,5]+d[4,1]
      flip = np.unravel_index(np.argmin(c),c.shape)
      
      for i in range(3):
         if flip[i] == 1:
            paths[i,:,:] = np.flipud(paths[i,:,:])
            dx[i,:] = np.flipud(dx[i,:])
            
      # choose point further from midpoint
      for (i,j) in [(0,1), (1,2), (2,0)]:
         s1 = midpoint-paths[i,3,:]
         s2 = midpoint-paths[j,0,:]
         di = np.linalg.norm(s1)
         dj = np.linalg.norm(s2)
         if di > dj:
            paths[j,0,:] = paths[i,3,:]
         else:
            paths[i,3,:] = paths[j,0,:]
         
         if angleTooWide(dx[i,1],dx[j,0],allowedAnkle):
            badTriple = True
            break
            
      if badTriple: continue
      
      # determine new path for curve
      anker = paths[0,0,:]
      newPath = "M{0[0]},{0[1]}".format(anker)
      for i in range(3):
         for j in range(1,4):
            num = "{0[0]},{0[1]}".format(paths[i,j,:]-paths[i,0,:])
            newPath = newPath + (" c" if (j == 1) else " ") + num
      pathAttributes["d"] = newPath
      
      # create new node for calculated polybezier curve and delete old nodes
      cf = et.SubElement(layer, "ns0:g", attrib={"id":"cuneiform"+str(cfnr)})
      newel = et.SubElement(cf, "ns0:path", attrib=pathAttributes)
   
      for curvenr in triple:
         g = layer.find(".//*[@curvenr='"+str(idx[curvenr])+"']")
         layer.remove(g)
         unUsedCurves[curvenr] = 0
   
      maxDist = max(np.max(dist[triple,:][:,triple]), maxDist)   
      cfnr = cfnr+1
      
   print cfnr, "cuneiforms found" 
   return maxDist, idx[np.squeeze(np.nonzero(unUsedCurves))]


# get the distances of the reference points to each other and sort them horizontally
# note: the closest point is always the currently considered point itself
dist = squareform(pdist(refPoints))

max1, unUsed = meltPaths(absPoints, refPoints, derivatives, dist, np.array(range(curvenr)))
max2, idx = meltPaths(absPoints, refPoints, derivatives, dist, unUsed)
maxTotal = max(max1,max2)

closePoints = [x for x in np.array(np.where(dist[idx,:][:,idx] < maxTotal)).T if x[0]<x[1] ]

# look for pairs
k=0
pairings = [[] for i in range(len(idx))]
for pair in closePoints:
   paths = absPoints[idx[pair,:],:,:]
   midpoint = np.mean(refPoints[idx[pair,:],:],axis=0)
   slope = paths[:,::3,:]-midpoint
   
   c = np.empty((2,2))
   c[0,0] = getAngle(slope[0][1],slope[1][0])
   c[0,1] = getAngle(slope[0][1],slope[1][1])
   c[1,0] = getAngle(slope[0][0],slope[1][0])
   c[1,1] = getAngle(slope[0][0],slope[1][1])
   
   flip = np.unravel_index(np.argmin(c),c.shape)

   if angleTooWide(np.min(c),0,0.1): continue
   
   k = k+1
   pairings[pair[0]].append(pair[1])
   

triples = []
for k in range(len(idx)):
   if len(pairings[k]) > 1:
      for (i,j) in combinations(pairings[k],2):
         if j in pairings[i]:
            triple = (idx[k],idx[i],idx[j])
            triples.append(triple)
            
   
   
maxDist = 0
unUsedCurves = np.ones(len(idx))
cfnr=0
# calculate one path for each group of three paths
for triple in triples:
   badTriple = False
   paths = absPoints[triple,:,:]
   dx = derivatives[triple,:]
   midpoint = np.mean(refPoints[triple,:],axis=0)
   
   # determine which curves have to be turned around
   c = np.empty((2,2,2)) 
   d = squareform(pdist(dx.reshape(6,1)))
   
   c[0,0,0] = d[1,2]+d[3,4]+d[5,0]
   c[0,0,1] = d[1,2]+d[3,5]+d[4,0]
   c[0,1,0] = d[1,3]+d[2,4]+d[5,0]
   c[0,1,1] = d[1,3]+d[2,5]+d[4,0]
   c[1,0,0] = d[0,2]+d[3,4]+d[5,1]
   c[1,0,1] = d[0,2]+d[3,5]+d[4,1]
   c[1,1,0] = d[0,3]+d[2,4]+d[5,1]
   c[1,1,1] = d[0,3]+d[2,5]+d[4,1]
   flip = np.unravel_index(np.argmin(c),c.shape)
   
   
   for i in range(3):
      if flip[i] == 1:
         paths[i,:,:] = np.flipud(paths[i,:,:])
         dx[i,:] = np.flipud(dx[i,:])
         
   # choose point further from midpoint
   for (i,j) in [(0,1), (1,2), (2,0)]:
      di = np.linalg.norm(midpoint-paths[i,3,:])
      dj = np.linalg.norm(midpoint-paths[j,0,:])
      if di > dj:
         paths[j,0,:] = paths[i,3,:]
      else:
         paths[i,3,:] = paths[j,0,:]
      
      if angleTooWide(dx[i,1],dx[j,0],allowedAnkle):
         badTriple = True
         break
         
   if badTriple: continue
   
   # determine new path for curve
   anker = paths[0,0,:]
   newPath = "M{0[0]},{0[1]}".format(anker)
   for i in range(3):
      for j in range(1,4):
         num = "{0[0]},{0[1]}".format(paths[i,j,:]-paths[i,0,:])
         newPath = newPath + (" c" if (j == 1) else " ") + num
   pathAttributes["d"] = newPath
   
   # create new node for calculated polybezier curve and delete old nodes
   cf = et.SubElement(layer, "ns0:g", attrib={"id":"cuneiform"+str(cfnr)})
   newel = et.SubElement(cf, "ns0:path", attrib=pathAttributes)

   for curvenr in triple:
      g = layer.find(".//*[@curvenr='"+str(curvenr)+"']")
      layer.remove(g)

   maxDist = max(np.max(dist[triple,:][:,triple]), maxDist)   
   cfnr = cfnr+1
   
print cfnr, "cuneiforms found" 


# Save changed svg file
tree.write(outputfile)




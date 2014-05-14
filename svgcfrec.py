import sys, optparse, time
import xml.etree.ElementTree as et
import re
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.optimize import minimize
import collections as col
from itertools import combinations, imap, ifilter, permutations, takewhile, dropwhile, groupby
import itertools
from pylab import plot,show
import warnings
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.misc import comb

def cubicBezier(points, t):
   xPoints = np.array([p[0] for p in points])
   yPoints = np.array([p[1] for p in points])
   polynomial_array = np.array([comb(3, i) * t**(3-i) * (1 - t)**i for i in range(0, 4)])
   xvals = np.dot(xPoints, polynomial_array)
   yvals = np.dot(yPoints, polynomial_array)
   return np.column_stack((xvals,yvals))

def getReferencePoint(points):
	average = np.mean(points[:,1:3,:],axis=1)
	try:
		# get intersection of lines defined by
		# - point 0 and 1 and
		# - point 2 and 3 respectively:
		A = np.vstack((points[1]-points[0], points[2]-points[3])).T
		x = np.linalg.solve(A,(points[2]-points[0]).T)
		intersection = points[0] + x[0] * points[1]
		pathlength = reduce(lambda x,y: x + np.linalg.norm(y), points[:,0:3,:] - points[:,1:4,:], 0.0)
		if np.linalg.norm(intersection - average) > pathlength:
			refPoint = average
		else:
			refPoint = intersection
	except np.linalg.LinAlgError: # when lines are parallel
		refPoint = average
	return refPoint


def getAngle(v1,v2):
   # return angle between vector v1 and vector v2
   np.seterr(invalid='ignore')
   nominator = np.linalg.norm(v1,2)*np.linalg.norm(v2,2)
   if nominator <1e-7: return 0.0
   res = np.arccos(np.dot(v1,v2)/nominator)
   if np.isnan(res) or np.isinf(res):
      return 0.0
   else:
      return res


def angleTooWide(angle,allowedAngle):
   # determine if an angle is too wide
   return abs(abs(angle)-(np.pi/2)) < (np.pi/2)-allowedAngle
   #return abs(angle-(np.pi/2)) < (np.pi/2)-allowedAngle


"""
Update element tree: delete old curves and create new node with
concatenated curve
"""
def updatePath(layer, paths, curveNumbers, groupID):
   # determine new path for curve
   paths = np.around(paths,decimals=3)
   anker = paths[0,0,:]

   newPath = "M{0[0]},{0[1]}".format(anker)
   for i in range(len(paths)):
      for j in range(1,4):
         newPath = newPath + (" c" if (j == 1) else " ")  \
                  + "{0[0]},{0[1]}".format(paths[i,j,:]-paths[i,0,:])
   pathAttributes["d"] = newPath

   # create new node for calculated polybezier curve and delete old nodes
   cf = et.SubElement(layer, "ns0:g", attrib={"id":"gshhrah"})
   newel = et.SubElement(cf, "ns0:path", attrib=pathAttributes)

   #for curvenr in curveNumbers:
    #  g = layer.find(".//*[@curvenr='"+str(curvenr)+"']")
    #  layer.remove(g)
   return True

def curveLineIntersections(bezier, line):
   a = -1*bezier[0,:] + 3*bezier[1,:] - 3*bezier[2,:] + 1*bezier[3,:]
   b = 3*bezier[0,:] - 6*bezier[1,:] + 3*bezier[2,:]
   c = -3*bezier[0,:] + 3*bezier[1,:]
   d = bezier[0,:]
   A = (line[1,1]-line[0,1])*a[0] + (line[0,0]-line[1,0])*a[1]
   B = (line[1,1]-line[0,1])*b[0] + (line[0,0]-line[1,0])*b[1]
   C = (line[1,1]-line[0,1])*c[0] + (line[0,0]-line[1,0])*c[1]
   D = (line[1,1]-line[0,1])*(d[0]-line[0,0]) + (line[0,0]-line[1,0])*(d[1]-line[0,1])
   ts = [x.real for x in np.roots([A,B,C,D]) if x.imag<1e-8]
   return np.column_stack((np.polyval([a[0],b[0],c[0],d[0]], ts),np.polyval([a[1],b[1],c[1],d[1]], ts)))


def valid(flippedPaths, gapSize):
   lines = flippedPaths[:,::3,:]
   for i in range(3):
      line2isecDists = [np.linalg.norm(x-lines[i-1,1,:]) for x in curveLineIntersections(flippedPaths[i,:,:],lines[i-1,:,:])] \
                        + [np.linalg.norm(x-lines[i,0,:]) for x in curveLineIntersections(flippedPaths[i-1,:,:],lines[i,:,:])]
      if len(line2isecDists) ==0 or min(line2isecDists) > gapSize: return False
   return True

def evalLine(start, end, n):
   return np.vstack(map(lambda i: start + (i/(n-1.0))*(end-start),range(n)))
   
def catPaths(absPoints, endSlopes, refPoints, triples, idx, \
                                    wedgeNrOffset, dist, lines):#, lineidx):
   # calculate one path for each group of three paths,
   unUsedCurves = np.ones(len(idx))
   maxCurveDist = 0
   for triple in triples:
      paths = absPoints[idx[triple,:],:,:]
      dx = endSlopes[idx[triple,:],:]
      midpoint = np.mean(refPoints[idx[triple,:],:],axis=0)

      """
      Check if a smooth transition between paths is possible, if not, proceed
      with next triple, else calculate it
      """

      # determine of which curves the indices have to be flipped
      # for melting the paths into one
      d = squareform(pdist(paths[:,::3,:].reshape(6,2)))
      #i1 = range(0,6,2)
      #i2 = range(1,6,2)
      #d[i1+i2,i2+i1] = 0.0
      #print d
      #perm = []
      #cost =[]
      #for i in itertools.permutations(range(6),6):
         #perm.append(i)
         #cost.append(np.sum(d[i[1:],i[:-1]]))
      #print cost
      #print perm
      #k = np.argmin(cost)
      #print cost[k], perm[k]

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
      #sys.exit()
      for i in range(3):
         if flip[i] == 1:
            paths[i,:,:] = np.flipud(paths[i,:,:])
            dx[i,:] = np.flipud(dx[i,:])

      if valid(paths,2.0) == False: continue
      
      # for two adjoining curves from the points where they meet
      # choose the point further from midpoint
      badTriple = False
      for (i,j) in [(0,1), (1,2), (2,0)]:
         s1 = midpoint - paths[i,3,:]
         s2 = midpoint - paths[j,0,:]
         if True:#getAngle(s1,s2) < allowedAngle: # leave in?
            di = np.linalg.norm(s1)
            dj = np.linalg.norm(s2)
            if di > dj:
               if abs(di-dj) > 1.0:
                  curve = np.vstack(map(lambda t: cubicBezier(paths[j],t), np.linspace(1.0, 0.0,10)))
                  line = evalLine(paths[i,3,:],paths[j,0,:],4)
                  paths[j,:,:] = bezierFit(np.vstack((line,curve[1:])))
               else: 
                  paths[j,0,:] = paths[i,3,:]
            else:
               if abs(di-dj) > 1.0:
                  curve = np.vstack(map(lambda t: cubicBezier(paths[i],t), np.linspace(1.0, 0.0,4)))
                  line = evalLine(paths[i,3,:],paths[j,0,:],4)
                  paths[i,:,:] = bezierFit(np.vstack((curve[:-1],line)))
               else:  paths[i,3,:] = paths[j,0,:]
         else:
            badTriple = True
            break

      if badTriple == True: continue

      #print paths
      #print valid(paths,1.0)
      #if valid(paths,1.0) == False: continue

      
      if options.extension:
         findExtension(paths,lines,midpoint)

      tripleArray = np.array(triple)

      updatePath(layer, paths, idx[tripleArray],"wedge"+str(wedgeNrOffset))
      unUsedCurves[tripleArray] = 0
      maxCurveDist = max(np.max(dist[triple,:][:,triple]), maxCurveDist)
      wedgeNrOffset = wedgeNrOffset+1

   # calculate new index mapping for free curves
   idx = idx[np.squeeze(np.nonzero(unUsedCurves))]
   return wedgeNrOffset, idx, maxCurveDist

def getPoints(pathNode):
   it = re.finditer('([MmCcSsLl])([^A-DF-Za-df-z]+)',pathNode.get("d").replace("-"," -").replace("e -", "e-"))
   cOffset = [0.0,0.0]
   points = []
   for m in it:
      char = m.group(1)
      pts = filter(lambda x: len(x)!=0, re.split(' |,', m.group(2).strip()))
      if char == "M":
         if len(pts)%2 != 0: return np.array([])
         cOffset = [float(pts[0]),float(pts[1])]
         points.append(cOffset)
         for i in range(2, len(points),2):
            cOffset = [float(pts[i]),float(pts[i+1])]
            points.append(cOffset)
            points.append(cOffset)
            points.append(cOffset)
      elif char == "m":
         if len(pts)%2 != 0: return np.array([])
         cOffset = [float(pts[0]),float(pts[1])]
         points.append(cOffset)
         for i in range(2,len(points),2):
            cOffset = [float(pts[i])+cOffset[0],float(pts[i+1])+cOffset[1]]
            points.append(cOffset)
            points.append(cOffset)
            points.append(cOffset)
      elif char == "C":
         if len(pts)%6 != 0: return np.array([])
         for i in range(0,len(pts),2):
            points.append([float(pts[i]),float(pts[i+1])])
      elif char == "c":
         if len(pts)%6 != 0: return np.array([])
         for i in range(0,len(pts),2):
            pt = [float(pts[i])+cOffset[0],float(pts[i+1])+cOffset[1]]
            points.append(pt)
            if i%3 == 1: cOffset = pt
      elif char == "S":
         if len(pts)%4 != 0: return np.array([])
         for i in range(0,len(pts),2):
            pt = [float(pts[i]),float(pts[i+1])]
            if i%4 == 0:
               points.append(np.add(np.subtract(points[-1],points[-2]),points[-1]))
               points.append(pt)
            else:
               points.append(pt)
               cOffset = pt
      elif char == "s":
         if len(pts)%4 != 0: return np.array([])
         for i in range(0,len(pts),2):
            pt = [float(pts[i])+cOffset[0],float(pts[i+1])+cOffset[1]]
            if i%4 == 0:
               points.append(np.add(np.subtract(points[-1],points[-2]),points[-1]))
               points.append(pt)
            else:
               points.append(pt)
               cOffset = pt
      elif char == "L":
         if len(pts)%2 != 0: return np.array([])
         for i in range(0, len(pts),2):
            cOffset = [float(pts[i]),float(pts[i+1])]
            points.append(cOffset)
            points.append(cOffset)
            points.append(cOffset)
      elif char == "l":
         if len(pts)%2 != 0: return np.array([])
         for i in range(0, len(pts),2):
            cOffset = [float(pts[i])+cOffset[0],float(pts[i+1])+cOffset[1]]
            points.append(cOffset)
            points.append(cOffset)
            points.append(cOffset)
      else: return np.array([])
   return np.array(points, dtype=floatType)

"""
Fit ordered points to a cubic bezier using least squares method
Takes n*3+1 points, returns 4
"""
def bezierFit(pts):
   pathlengths = [0.0]
   for i in range(len(pts)-1):
      pathlengths.append(pathlengths[i] + np.linalg.norm(pts[i,:]-pts[i+1,:]))
   ts = map(lambda x: x / pathlengths[-1], pathlengths)
   x0 = np.array(map(lambda t: (1.0-t)**3, ts ))
   x1 = np.array(map(lambda t: 3*(1.0-t)**2*t, ts ))
   x2 = np.array(map(lambda t: 3*(1.0-t)*t**2, ts ))
   x3 = np.array(map(lambda t: t**3, ts ))
   X = np.column_stack((x0,x1,x2,x3))
   Cs = np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T,pts))
   return Cs

def separator(first, last, p, sw):
   if np.linalg.norm(first-p) < sw:
      return 0
   if np.linalg.norm(last-p) < sw:
      return 2
   else: return 1

def thinOut(points):
   data = []
   step = np.linspace(1,0,4,False)
   
   for i in range(0,len(points)-3,3):
      data.extend(cubicBezier(points[i:i+4,:],step))
   data.extend(np.array([points[-1,:]]))
   
   sw = 0.51
   first = data[0]
   last = data[np.argmax(cdist(data,[first]))]

   # groups the data: 0) points near first 1) points to last 2) points near last 3) [points to first 4) points near first ]
   order = ''
   pts = []
   for k,l in groupby(data, key=lambda p: separator(first, last, p, sw)):
      order = order + str(k)
      pts.append(np.array(list(l)))

   #sys.exit()
   if order == '02': # line
      p0 = np.mean(pts[0],axis=0)
      p2 = np.mean(pts[1],axis=0)
      controlPoints = np.vstack((p0, p0, p2, p2))
   if order == '012':
      curve = np.vstack((np.mean(pts[0],axis=0),pts[1], np.mean(pts[2],axis=0)))
      controlPoints = bezierFit(curve)
   elif order == '0120':
      curve = np.vstack(np.mean(np.vstack((pts[0],pts[3])),axis=0), pts[1],np.mean(pts[2],axis=0))
      controlPoints = bezierFit(curve)
   elif order == '0121':
      controlPointsA = bezierFit(np.vstack(pts[0][-1,:], pts[1], pts[2][0,:]))
      controlPointsB = bezierFit(np.vstack(pts[0][0,:], pts[3][::-1], pts[2][-1,:]))
      controlPoints = np.mean([controlPointsA, controlPointsB],axis=0)
   elif order == '01210':
      controlPointsA = bezierFit(np.vstack(pts[0][-1,:], pts[1], pts[2][0,:]))
      controlPointsB = bezierFit(np.vstack(pts[4][0,:], pts[3][::-1], pts[2][-1,:]))
      controlPoints = np.mean([controlPointsA, controlPointsB],axis=0)
   else:
      return None

   return controlPoints

def transform(points, transformation):
   if transformation == None: return points
   m = re.match(r"(?P<method>\w+)\(\s*(?P<args>[^)]*)\)", transformation)
   args = m.group('args').replace(",", " ").split()
   if m.group('method') == "translate" and (len(args) < 3 or len(args) > 0):
      points[:,0] = points[:,0] + np.float(args[0])
      if len(args) == 2: points[:,1] = points[:,1] + np.float(args[1])
   else:
      if options.verbose:
         print "Invalid transform operation", m.group('method')
         print "Using original points..."
   return points

def adjust(points, transformation):
   if len(points) >= 4:
      points = transform(points, transformation)
      if len(points) == 4:       return points
      elif options.polybezier:   return thinOut(points)
   return None


"""
Find lines, for they could be an extension to wedge
"""
def getLines(layer, ns, offset=0):
   lines = []
   if options.verbose: print "\nFinding lines..."

   for g in layer:
      # get points
      if g.tag == ns + "line":
         g.set("linenr",str(len(lines)+offset))
         lines.append( [[g.get("x1"), g.get("y1")], \
                        [g.get("x2"), g.get("y2")]] )

   if options.verbose:  print len(lines), "lines found"

   return np.array(lines,dtype=floatType).reshape((len(lines),2,2))

"""
Find curves which could belong to the wedge
"""
def getPaths(layer, ns):
   curves = np.empty([len(layer),4,2],dtype=floatType)
   lines = np.empty([len(layer),2,2],dtype=floatType)
   if options.verbose: print "\nFinding cubic bezier curves..."

   # assumes each child of layer representing exactly one curve, which is
   # defined by the first path within this subtree
   curvenr = 0
   linenr = 0
   nonConformCurves = 0
   for g in layer:
      # get points
      path = g.find(".//"+ns+"path")
      if path != None:
         points = adjust(getPoints(path), g.get("transform"))
         #print "paths", points

         if points != None:

            # check if the curve is not more like a line
            pathlength = np.linalg.norm(points[:-1,:]-points[1:,:],2)
            linearDistance = np.linalg.norm(points[0,:]-points[3,:],2)

            if linearDistance < 0.5: # point
               nonConformCurves = nonConformCurves + 1
            elif abs(pathlength - linearDistance)/linearDistance < lineThreshold:
               lines[linenr] = points[(0,3),:]
               g.set("linenr",str(linenr))
               linenr = linenr + 1
            else:
               curves[curvenr] = points
               g.set("curvenr",str(curvenr))
               curvenr = curvenr + 1

         else: nonConformCurves = nonConformCurves + 1

   if options.verbose:
      print nonConformCurves + curvenr + linenr, "curves found"
      print " ", curvenr, "curves with correct formatting"
      print " ", linenr, "curves classified as line"

   # trim to real size
   curves.resize((curvenr,4,2))
   lines.resize((linenr,2,2))

   return curves, lines

def updateLine(layer, points, lineNumbers, lineID):

   # set attributes
   lineAttributes["x1"] = str(points[0,0])
   lineAttributes["y1"] = str(points[0,1])
   lineAttributes["x2"] = str(points[1,0])
   lineAttributes["y2"] = str(points[1,1])
   lineAttributes["id"] = lineID

   # create new node for calculated polybezier curve and delete old nodes
   cf = et.SubElement(layer, "ns0:line", attrib=lineAttributes)

   #for linenr in lineNumbers:
    #  g = layer.find(".//*[@linenr='"+str(linenr)+"']")
    #  layer.remove(g)
   return

def debug(layer):
   nsPrefix = layer.tag[:layer.tag.find("}")+1]
   curves, lines = getPaths(layer,nsPrefix)
   lines = np.vstack((lines, getLines(layer,nsPrefix,len(lines))))

   for nr in range(len(curves)):
      if curves[nr] != None:
         updatePath(layer,[curves[nr]],[nr],"curve"+str(nr))
      else: print "No curves"

   for nr in range(len(lines)):
      if lines[nr] != None:
         updateLine(layer,lines[nr],[nr],"line"+str(nr))
      else: print "No lines"

   return

def findExtension(paths, lines, midpoint):
   pathCorners = paths[:,3,:].reshape(len(paths),2)

   #sys.exit()
   lineEdges = lines.reshape((len(lines)*2,2))
   #print paths
   (pCorner,lEdge) = np.where(cdist(pathCorners,lineEdges) < extensionDist)

   print paths
   print pathCorners
   for i in range(len(pCorner)):
      start = lEdge[i]%2
      angle = getAngle( midpoint - lines[lEdge[i]/2,1-start,:],
                        midpoint - paths[pCorner[i],3,:]   )
      if angle > extensionAngle:
         #print "angle:", angle, extensionAngle
         #print getAngle([1,0],[0,1])
         return paths

      paths[pCorner[i],3,:] = lines[lEdge[i]/2,1-lEdge[i]%2,:]
      paths[pCorner[i]-2,0,:] = paths[pCorner[i],3,:]
   #print paths
   #sys.exit()
   return paths

"""
Calculate the distances from the reference points to each other, find groups
of three curves belonging together and calculate concatenated paths
"""
def update(layer):
   nsPrefix = layer.tag[:layer.tag.find("}")+1]

   absPoints, lines = getPaths(layer,nsPrefix)
   lines = np.vstack((lines, getLines(layer,nsPrefix,len(lines))))

   nCurves = len(absPoints)

   # get reference point and derivatives at end points for curve
   #endSlopes = np.empty([nCurves,2,2],dtype=floatType)
   endSlopes = np.dstack(( absPoints[:,1,:] - absPoints[:,0,:], \
                           absPoints[:,2,:] - absPoints[:,3,:]  ))

   #print absPoints, endSlopes, endSlopes.shape

   #refPoints = np.mean(absPoints[:,1:3,:], axis=1)
   refPoints = getReferencePoint(absPoints)

   #print absPoints[0], refPoints[0], endSlopes[0]
   #print absPoints.shape, refPoints.shape, endSlopes.shape
   if nCurves == 0:
      if options.verbose:
         print "\nNo suitable curves found in layer", options.layerID
         print "Program aborting..."
      sys.exit(0)

   if options.verbose: print "\nExecuting first strategy..."
   dist = squareform(pdist(refPoints))
   maxCurveDist = 0

   cfnrOld = 0
   cfnr = 0
   idx = np.array(range(nCurves))
   run = 1

   while True:
      # return only index triples that appear three times
      # (meaning: for each reference point in the tuple the same
      # group of three including itself is found)
      # note: cast to "tuple" makes the entries hashable as required
      closest = np.argsort(dist[idx,:][:,idx])
      count = col.Counter([tuple(x) for x in np.sort(closest[:,:3])])
      cuneiforms = [k for (k, v) in count.items() if v==3]
		# note: versuche alle tripel zu testen (itertools.combinations)
      #print len(cuneiforms)
      cfnr, idx, maxD = catPaths(absPoints, endSlopes, refPoints, cuneiforms, idx, cfnr, dist, lines)

      if cfnrOld != cfnr:
         if options.verbose: print cfnr, "cuneiforms found after run", run
         run = run + 1
         maxCurveDist = max(maxCurveDist,maxD)
         cfnrOld = cfnr
      else: break

   if options.verbose:
      print "\n", len(idx), "curves left"
      print "\nExecuting second strategy..."


   # look for pairs of curves which are not further away from each other than
   # the pairs in the cuneiforms already found

   #"""
   closePoints = filter(lambda x: x[0]<x[1], np.array(np.where(dist[idx,:][:,idx] < maxCurveDist)).T)

   #print closePoints
   #sys.exit()
   
   print maxCurveDist, len(closePoints)
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

      #print paths

      flip = np.unravel_index(np.argmin(c),c.shape)

      if angleTooWide(np.min(c),1.0): continue

      k = k+1
      pairings[pair[0]].append(pair[1])

   triples = []
   used = set([])
   for k in range(len(idx)):
      if len(pairings[k]) > 1:
         for (i,j) in combinations(pairings[k],2):
            if j in pairings[i] and \
               k not in used and \
               i not in used and \
               j not in used:

               triple = (k,i,j)
               triples.append(triple)
               used.add(k)
               used.add(i)
               used.add(j)
   print len(triples)/3

   cfnr, idx, maxD = catPaths(absPoints, endSlopes, refPoints, triples, idx, cfnr, dist, lines)
   #"""
   if options.verbose: print cfnr, "cuneiforms found in total\n"

if __name__ == '__main__':
    pathAttributes = {"fill":"none", "stroke": "blue", "stroke-width": "0.5"} #"d":"",
    lineAttributes = {"fill":"none", "stroke": "yellow", "stroke-width": "0.5"}
    floatType = np.float32
    allowedAngle = np.pi/3 # 5 #10
    lineThreshold = 1e-7
    extensionAngle = np.pi/15
    extensionDist = 1
    pts = np.array([[259.251, 290.61199999999997], [261.334, 284.778], [264.001, 280.61199999999997], [265.084, 278.86199999999997], [266.417, 277.27799999999996]])
    #print bezierFit(pts, np.array([254.167, 292.445]), np.array([268.001, 276.11199999999997]))
    #sys.exit()
    try:
        start_time = time.time()
        parser = optparse.OptionParser(formatter=optparse.TitledHelpFormatter(), usage=globals()['__doc__'], version='$Id$')
        parser.add_option ('-v', '--verbose', action='store_true', default=False, help='verbose output')
        parser.add_option ('-p', '--polybezier', action='store_true', default=False, help='Include polybeziers (find best fit)')
        parser.add_option ('-e', '--extension', action='store_true', default=False, help='Consider lines for wedge extensions')
        parser.add_option ('-o', '--ofile', metavar="FILE", action='store', default="out.svg", help='output file', dest="outputfile")
        parser.add_option ('-l', '--layerid', metavar="ID", action='store', default="cuneiforms", help='layer ID', dest="layerID")
        (options, args) = parser.parse_args()
        if options.verbose: print time.asctime()
        if len(args) < 1:
            #print "No input file given"
            #sys.exit(0)
            args.append("test/test3.svg")

        if options.verbose: print "\nInput: ", args[0]

        tree = et.parse(args[0])
        root = tree.getroot()
        layer = root.find(".//*[@id='"+options.layerID+"']")
        if layer == None:
            print "No layer with id='"+options.layerID+"' found"
            print "Program aborting..."
            sys.exit(0)

        #debug(layer)
        update(layer)

        if options.verbose: print "Writing result to: ", options.outputfile, "\n"
        tree.write(options.outputfile)

        if options.verbose: print time.asctime()
        if options.verbose: print 'TOTAL TIME IN SECONDS:',
        if options.verbose: print (time.time() - start_time)
        sys.exit(0)
    except KeyboardInterrupt, e:
        raise e
    except SystemExit, e:
        raise e

"""
python svgcfrec.py VAT_10908_Vs.svg -v -l Kopie
python svgcfrec.py VAT_10321_Vs_SJakob.svg -v -l Autographie -p
python svgcfrec.py VAT_09671_Rs_SJakob.svg -v -l Autographie -p
python svgcfrec.py VAT_09898+10964_Vs_SJakob.svg -v -l Autographie
python svgcfrec.py VAT_10321_Vs_SJakob.svg -v -l Autographie
python svgcfrec.py VAT_11022_SJakob.svg -v -l Autographie
python svgcfrec.py VAT_10622_HPSchaudig.svg -v -l g20
python svgcfrec.py VAT_10686+Obv_HPSchaudig.svg -v -l g20
python svgcfrec.py VAT_10833-SeiteB_HPSchaudig.svg -v -l g20
"""

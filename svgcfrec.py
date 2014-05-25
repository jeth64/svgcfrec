import numpy as np
from sys                    import exit
from time                   import time, asctime
from optparse               import OptionParser, TitledHelpFormatter
from xml.etree.ElementTree  import parse, SubElement
from re                     import finditer, split, match
from scipy.spatial.distance import pdist, squareform, cdist
from collections            import Counter
from itertools              import combinations, groupby, product, repeat
from scipy.misc             import comb
from operator               import add


def cubicBezier(points, t):
   xPoints = np.array([p[0] for p in points])
   yPoints = np.array([p[1] for p in points])
   polynomial_array = np.array([comb(3, i) * t**(3-i) * (1 - t)**i for i in range(0, 4)])
   xvals = np.dot(xPoints, polynomial_array)
   yvals = np.dot(yPoints, polynomial_array)
   return np.column_stack((xvals,yvals))

def getReferencePoint(points):
   average = np.mean(points,axis=0)
   try:
      # get intersection of lines defined by
      # - point 0 and 1 and
      # - point 2 and 3 respectively:
      A = np.vstack((points[1]-points[0], points[2]-points[3])).T
      x = np.linalg.solve(A,(points[2]-points[0]).T)
      intersection = points[0] + x[0] * (points[1]-points[0])
      pathlength = reduce(lambda x,y: x + np.linalg.norm(y), points[0:3,:] -points[1:4,:], 0.0)
      if np.linalg.norm(intersection - average) > pathlength:
         refPoint = average
      else:
         refPoint = intersection
   except np.linalg.LinAlgError: # when lines are parallel
      refPoint = average
   return refPoint


"""
Update element tree: delete old curves and create new node with concatenated curve
"""
def updatePath(layer, paths, curveNumbers, lineNumbers, groupID):
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
   cf = SubElement(layer, "ns0:g", attrib={"id":groupID})
   newel = SubElement(cf, "ns0:path", attrib=pathAttributes)

   if options.delete:
      for nr in curveNumbers:
         node = layer.find(".//*[@curvenr='"+str(nr)+"']")
         if node != None: layer.remove(node)
      for nr in lineNumbers:
         node = layer.find(".//*[@linenr='"+str(nr)+"']")
         if node != None: layer.remove(node)
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
   for i in range(len(flippedPaths)):
      line2isecDists = [np.linalg.norm(x-lines[i-1,1,:]) for x in curveLineIntersections(flippedPaths[i,:,:],lines[i-1,:,:])] \
                        + [np.linalg.norm(x-lines[i,0,:]) for x in curveLineIntersections(flippedPaths[i-1,:,:],lines[i,:,:])]
      if len(line2isecDists) ==0 or min(line2isecDists) > gapSize: return False
   return True
   
def evalLine(start, end, n):
   return np.vstack(map(lambda i: start + (i/(n-1.0))*(end-start),range(n)))

"""
For two adjoining curves from the points where they meet
choose the point further from midpoint
"""
def mergeEnds(paths, midpoint):
   for (i,j) in [(k-1,k) for k in range(len(paths))]:
      di = np.linalg.norm(midpoint - paths[i,3,:])
      dj = np.linalg.norm(midpoint - paths[j,0,:])
      if di > dj:
         paths[j,0,:] = paths[i,3,:]
      else:
         paths[i,3,:] = paths[j,0,:]
   return paths

"""
Determine of which curves the indices have to be flipped
for melting the paths into one
"""
def flipCurves(paths):
   func = lambda p: reduce(add,[np.linalg.norm(p[k-1][3,:]-p[k][0,:]) for k in range(len(p))])
   possibilities = map(lambda x: map(lambda a, b, c: b if a else c, x, paths, paths[:,::-1,:]), product([True, False], repeat=3))
   return np.array(min(possibilities, key=func))


def catPaths(absPoints, endSlopes, refPoints, triples, idx, \
                                    wedgeNrOffset, dist, lines, lineidx):
   # calculate one path for each group of three paths,
   unUsedCurves = np.ones(len(idx))
   unUsedLines = np.ones(len(lineidx))
   maxCurveDist = 0
   for triple in triples:
      paths = absPoints[idx[triple,:],:,:]
      dx = endSlopes[idx[triple,:],:]
      midpoint = np.mean(refPoints[idx[triple,:],:],axis=0)

      """
      Check if a smooth transition between paths is possible, if not, proceed
      with next triple, else calculate it
      """

      paths = flipCurves(paths)
      if valid(paths,gapSize) == False:
         continue

      paths = mergeEnds(paths, midpoint)
      
      if options.extension:
         paths, usedLines = findExtension(paths,lines[lineidx],midpoint)

      tripleArray = np.array(triple)

      updatePath(layer, paths, idx[tripleArray], lineidx[usedLines], "wedge"+str(wedgeNrOffset))
      unUsedCurves[tripleArray] = 0
      unUsedLines[usedLines] = 0
      maxCurveDist = max(np.max(dist[triple,:][:,triple]), maxCurveDist)
      wedgeNrOffset = wedgeNrOffset+1

   # calculate new index mapping for free curves and lines
   idx = idx[np.squeeze(np.nonzero(unUsedCurves))]
   lineidx = lineidx[np.squeeze(np.nonzero(unUsedLines))]
   return wedgeNrOffset, idx, maxCurveDist, lineidx

def getPoints(pathstring):
   it = finditer('([MmCcSsLl])([^A-DF-Za-df-z]+)',pathstring.replace("-"," -").replace("e -", "e-"))
   cOffset = [0.0,0.0]
   points = []
   for m in it:
      char = m.group(1)
      pts = filter(lambda x: len(x)!=0, split(' |,', m.group(2).strip()))
      if char == "M":
         if len(pts)%2 != 0: return np.array([])
         cOffset = [float(pts[0]),float(pts[1])]
         points.append(cOffset)
         for i in range(2, len(pts),2):
            cOffset = [float(pts[i]),float(pts[i+1])]
            points.append(cOffset)
            points.append(cOffset)
            points.append(cOffset)
      elif char == "m":
         if len(pts)%2 != 0: return np.array([])
         cOffset = [float(pts[0]),float(pts[1])]
         points.append(cOffset)
         for i in range(2,len(pts),2):
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
      elif char == "L": # works? test8
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
      else:
         if options.verbose: print "\n Unknown character in path string detected"
         return np.array([])

   return np.array(points, dtype=floatType)

def elevation(qbezier):
   return np.array([qbezier[0],(qbezier[0]+2*qbezier[1])/3, \
                   (qbezier[2]+2*qbezier[1])/3, qbezier[2]])

                  
"""
Fit ordered points to a bezier of degree n using least squares method
Takes k*3+1 points, returns n+1
"""
def bezierFit(pts):
   pathlengths = [0.0]
   for i in range(len(pts)-1):
      pathlengths.append(pathlengths[i] + np.linalg.norm(pts[i,:]-pts[i+1,:]))
   ts = map(lambda x: x / pathlengths[-1], pathlengths)

   if len(pts) < 4:
      X = np.array(map(lambda t: [comb(2, i) * t**(i) * (1 - t)**(2-i) for i in range(0, 3)], ts))
      Cs = elevation(np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T,pts)))

   else:
      X = np.array(map(lambda t: [comb(3, i) * t**(i) * (1 - t)**(3-i) for i in range(0, 4)], ts))
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
   
   sw = float(options.strokewidth) + 0.05
   first = data[0]
   last = data[np.argmax(cdist(data,[first]))]

   # groups the data: 0) points near first 1) points to last 2) points near last 3) [points to first 4) points near first ]
   order = ''
   pts = []
   for k,l in groupby(data, key=lambda p: separator(first, last, p, sw)):
      order = order + str(k)
      pts.append(np.array(list(l)))

   if order == '02': # line
      p0 = np.mean(pts[0],axis=0)
      p2 = np.mean(pts[1],axis=0)
      controlPoints = np.vstack((p0, p0, p2, p2))
   elif order == '020': # line
      p0 = np.mean(np.vstack((pts[0],pts[2])),axis=0)
      p2 = np.mean(pts[1],axis=0)
      controlPoints = np.vstack((p0, p0, p2, p2))
   elif order == '012':
      curve = np.vstack((np.mean(pts[0],axis=0),pts[1], np.mean(pts[2],axis=0)))
      controlPoints = bezierFit(curve)
   elif order == '0120': # probably line
      curve = np.vstack((np.mean(np.vstack((pts[0],pts[3])),axis=0), pts[1],np.mean(pts[2],axis=0)))
      controlPoints = bezierFit(curve)
   elif order == '0121':
      controlPointsA = bezierFit(np.vstack((pts[0][-1,:], pts[1], pts[2][0,:])))
      controlPointsB = bezierFit(np.vstack((pts[0][0,:], pts[3][::-1], pts[2][-1,:])))
      controlPoints = np.mean([controlPointsA, controlPointsB],axis=0)
   elif order == '01210' or order == '0121010':
      controlPointsA = bezierFit(np.vstack((pts[0][-1,:], pts[1], pts[2][0,:])))
      controlPointsB = bezierFit(np.vstack((pts[4][0,:], pts[3][::-1], pts[2][-1,:])))
      controlPoints = np.mean([controlPointsA, controlPointsB],axis=0)
   elif order == '0210': # probably line
      curve = np.vstack((np.mean(np.vstack((pts[0],pts[3])),axis=0), pts[2][::-1], pts[1][-1,:]))
      controlPoints = bezierFit(curve)
   else:
      #if options.verbose: print "\n Unknown curve specification detected:", order
      return None
   return controlPoints

def transform(points, transformation):
   if transformation == None: return points
   m = match(r"(?P<method>\w+)\(\s*(?P<args>[^)]*)\)", transformation)
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

   if options.verbose: print " ", len(lines), "lines found"

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
         points = adjust(getPoints(path.get("d")), g.get("transform"))
         
         if points != None:

            # check if the curve is not more like a line
            pathlength = reduce(lambda x,y: x + np.linalg.norm(y), points[0:3,:] -points[1:4,:], 0.0)
             
            linearDistance = np.linalg.norm(points[0,:]-points[3,:],2)

            if linearDistance < float(options.strokewidth): # point
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
   cf = et.SubElement(layer, "ns0:line", attrib={"fill":"none", "stroke": "yellow", "stroke-width": "0.5"})

   if options.delete:
      for linenr in lineNumbers:
         g = layer.find(".//*[@linenr='"+str(linenr)+"']")
         layer.remove(g)
   return

def debug(layer):
   nsPrefix = layer.tag[:layer.tag.find("}")+1]
   curves, lines = getPaths(layer,nsPrefix)
   refPoints = np.array(map(lambda x: np.vstack((list(repeat(getReferencePoint(x),3)),getReferencePoint(x)+[0.5, 0.0])), curves))
   lines = np.vstack((lines, getLines(layer,nsPrefix,len(lines))))

   for nr in range(len(curves)):
      if curves[nr] != None:
         updatePath(layer,[curves[nr]],[nr],"curve"+str(nr))
      else: print "No curves"
   """
   for nr in range(len(refPoints)):
      if curves[nr] != None:
         updatePath(layer,[refPoints[nr]],[nr],"ref"+str(nr))
      else: print "No curves"
   """

   for nr in range(len(lines)):
      if lines[nr] != None:
         updateLine(layer,lines[nr],[nr],"line"+str(nr))
      else: print "No lines"

   return


def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0: return v
    return v/norm

def findExtension(paths, lines, midpoint):
   usedLines = []
   pathCorners = paths[:,0,:].reshape(len(paths),2)
   pCornerSlopes = [normalize(paths[:,0,:]-paths[:,1,:]), normalize(paths[range(-1,2),3,:]-paths[range(-1,2),2,:])]
   lineEdges = lines.reshape((len(lines)*2,2))
   (pCorner,lEdge) = np.where(cdist(pathCorners,lineEdges) < extensionDist)

   for i in range(len(pCorner)):
      lStart = lines[lEdge[i]/2,lEdge[i]%2,:]
      lEnd = lines[lEdge[i]/2,1-lEdge[i]%2,:]
      dirvec = normalize(lEnd-lStart)
      angle = np.dot( normalize(np.mean([pCornerSlopes[1][pCorner[i]],pCornerSlopes[0][pCorner[i]]],axis=0)), dirvec)

      if cosExtensionAngle < angle:
         paths[pCorner[i]-1,3,:] = lines[lEdge[i]/2,1-lEdge[i]%2,:]
         paths[pCorner[i],0,:] = paths[pCorner[i]-1,3,:]
         usedLines.append(lEdge[i]/2)
   return paths, usedLines


"""
Calculate the distances from the reference points to each other, find groups
of three curves belonging together and calculate concatenated paths
"""
def update(layer):
   nsPrefix = layer.tag[:layer.tag.find("}")+1]

   absPoints, lines = getPaths(layer,nsPrefix)
   lines = np.vstack((lines, getLines(layer,nsPrefix,len(lines))))

   nCurves = len(absPoints)
   nLines = len(lines)

   # get reference point and derivatives at end points for curve
   endSlopes = np.dstack(( absPoints[:,1,:] - absPoints[:,0,:], \
                           absPoints[:,2,:] - absPoints[:,3,:]  ))

   refPoints = np.array(map(getReferencePoint, absPoints))

   if nCurves == 0:
      if options.verbose:
         print "\nNo suitable curves found in layer", options.layerID
         print "Program aborting..."
      exit(0)

   if options.verbose: print "\nExecuting first strategy...\n"
   dist = squareform(pdist(refPoints))
   maxCurveDist = 0

   cfnrOld = 0
   cfnr = 0
   idx = np.array(range(nCurves))
   lineidx = np.array(range(nLines))
   run = 1

   while True:
      # return only index triples that appear three times
      # (meaning: for each reference point in the tuple the same
      # group of three including itself is found)
      # note: cast to "tuple" makes the entries hashable as required
      closest = np.argsort(dist[idx,:][:,idx])
      count = Counter([tuple(x) for x in np.sort(closest[:,:3])])
      cuneiforms = [k for (k, v) in count.items() if v==3]
      cfnr, idx, maxD, lineidx = catPaths(absPoints, endSlopes, refPoints, cuneiforms, idx, cfnr, dist, lines, lineidx)

      if cfnrOld != cfnr:
         if options.verbose: print cfnr, "wedges found after run", run
         run = run + 1
         maxCurveDist = max(maxCurveDist,maxD)
         cfnrOld = cfnr
      else: break

   if options.verbose:
      print "\n", len(idx), "curves left"
      print len(lineidx), "lines left"
      print "\nExecuting second strategy...\n"

   
   # look for pairs of curves which are not further away from each other than
   # the pairs in the cuneiforms already found

   if maxCurveDist > mDist: maxCurveDist= mDist

   closePoints = filter(lambda x: x[0]<x[1], np.array(np.where(dist[idx,:][:,idx] < maxCurveDist)).T)

   k=0
   pairings = [[] for i in range(len(idx))]
   for pair in closePoints:
      paths = absPoints[idx[pair,:],:,:]
      k = k+1
      pairings[pair[0]].append(pair[1])

   triples = []
   used = set([])
   for k in range(len(idx)):
      if len(pairings[k]) > 1:
         for (i,j) in combinations(pairings[k],2):
            if j in pairings[i] and k not in used and i not in used and j not in used:
               triple = (k,i,j)

               if valid(flipCurves(absPoints[triple,:,:]),gapSize):
                  triples.append(triple)
                  used.add(k)
                  used.add(i)
                  used.add(j)
   
   cfnr, idx, maxD, lineidx = catPaths(absPoints, endSlopes, refPoints, triples, idx, cfnr, dist, lines, lineidx)

   if options.verbose: print cfnr, "wedges found in total\n"

if __name__ == '__main__':
    floatType = np.float32
    
    try:
        start_time = time()
        parser = OptionParser(formatter=TitledHelpFormatter(), usage=globals()['__doc__'], version='$Id$')
        parser.add_option ('-v', '--verbose', action='store_true', default=False, help='verbose output')
        parser.add_option ('-p', '--polybezier', action='store_true', default=False, help='include polybeziers (find best fit)')
        parser.add_option ('-e', '--extension', action='store_true', default=False, help='consider lines for wedge extensions')
        parser.add_option ('-o', '--ofile', metavar="FILE", action='store', default="out.svg", help='output file', dest="outputfile")
        parser.add_option ('-l', '--layerid', metavar="ID", action='store', default="cuneiforms", help='layer ID', dest="layerID")
        parser.add_option ('-g', '--gapsize', metavar="SIZE", action='store', default=2.0, help='allowed gap between end point of of curve and intersection of curves', dest="gapSize")
        parser.add_option ('-t', '--linethreshold', metavar="THRESHOLD", action='store', default=1e-7, help='maximum distance of points to fitted line to be classified as such', dest="lineThreshold")
        parser.add_option ('-a', '--extangle', metavar="ANGLE", action='store', default=np.pi/5, help='maximum angle between extending line and direction vector of corner', dest="extAngle")
        parser.add_option ('-d', '--extDIST', metavar="DIST", action='store', default=1.0, help='distance between corner and line end', dest="extDist")
        parser.add_option ('-m', '--maxDist', metavar="DIST", action='store', default=1000.0, help='upper limit of the distance between reference points of the curves of a wedge', dest="maxCurveDist")
        
        parser.add_option ('-s', '--strokewidth', metavar="WIDTH", action='store', default="0.5", help='stroke-width of paths', dest="strokewidth")
        parser.add_option ('-c', '--strokecolor', metavar="COLOR", action='store', default="blue", help='stroke-color of paths', dest="strokecolor")

        parser.add_option ('', '--debug', action='store_true', default=False, help='only preparatory functions are executed; outfile contains detected curves and lines')
        parser.add_option ('', '--delete', action='store_true', default=False, help='delete used original curves from document')
        
        (options, args) = parser.parse_args()
        
        gapSize = float(options.gapSize)
        cosExtensionAngle = np.cos(float(options.extAngle))
        extensionDist = float(options.extDist)
        lineThreshold = float(options.lineThreshold)
        mDist = float(options.maxCurveDist)
        pathAttributes = {"fill":"none", "stroke": options.strokecolor, "stroke-width": options.strokewidth} 
        
        if options.verbose: print asctime()
        if len(args) < 1:
            print "No input file given"
            exit(0)

        if options.verbose: print "\nInput: ", args[0]

        tree = parse(args[0])
        layer = tree.getroot().find(".//*[@id='"+options.layerID+"']")
        if layer == None:
            print "No layer with id='"+options.layerID+"' found"
            print "Program aborting..."
            exit(0)

        if options.debug: debug(layer)
        else: update(layer)

        if options.verbose: print "Writing result to: ", options.outputfile, "\n"
        tree.write(options.outputfile)

        if options.verbose: print asctime()
        if options.verbose: print 'TOTAL TIME IN SECONDS:',
        if options.verbose: print (time() - start_time)
        exit(0)
    except KeyboardInterrupt, e:
        raise e
    except SystemExit, e:
        raise e

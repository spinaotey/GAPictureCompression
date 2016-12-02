import numpy as np
from PIL import Image
from PIL import ImageDraw

def boundedv(x,xmax):
    x[x<0] = 0;
    x[x>xmax] = xmax;

def bounded(x,xmax):
    return(xmax if x > xmax else (0 if x < 0 else x))

class PolyGen:
    bv = 0.001

    def randomPoint(self):
        point = np.empty((1,2),dtype='int32')
        point[0,0] = bounded(np.random.randint(-self.dx,self.xmax+1+self.dx,dtype='int32'),self.xmax)
        point[0,1] = bounded(np.random.randint(-self.dy,self.ymax+1+self.dy,dtype='int32'),self.ymax)
        return(point)

    def __init__(self,x,y,npol=3):
        self.n = np.int8(npol);
        self.coords = np.empty((npol,2),dtype='int32')
        self.xmax = np.int32(x)
        self.ymax = np.int32(y)
        self.dx = x*PolyGen.bv;
        self.coords[:,0] = np.random.randint(-self.dx,x+1+self.dx,npol,dtype='int32')
        boundedv(self.coords[:,0],x)
        self.dy = y*PolyGen.bv;
        self.coords[:,1] = np.random.randint(-self.dy,y+1+self.dy,npol,dtype='int32')
        boundedv(self.coords[:,1],y)
        self.color = np.random.randint(0,256,4,dtype='uint16')

    def mutatePoint1(self,i):
        self.coords[i,:] = self.randomPoint();

    def mutatePoint2(self,i):
        self.coords[i,:] += np.int32(np.random.normal(scale=min(self.xmax,self.ymax)/10,size=2))
        boundedv(self.coords[i,:],self.xmax)

    def mutateColor(self):
        self.color += np.uint16(np.random.normal(scale=256/15,size=4))
        self.color %= 256

    def mutateN(self):
        if self.n == 3:
            i = np.int8(1)
        elif self.n == 6:
            i = np.int8(-1)
        else:
            i = np.random.choice(np.int8([-1,1]))
        self.n += i
        if i == 1:
            self.coords = np.insert(self.coords,
                                    np.random.randint(self.n,size=1,dtype='int8'),
                                    self.randomPoint(),0)
        else:
            self.coords = np.delete(self.coords,
                                    np.random.randint(self.n,size=1,dtype='int8'),
                                    0)
                                    

    def makePoly(self):
        poly = Image.new('RGBA',(self.xmax,self.ymax),None)
        polydraw = ImageDraw.Draw(poly)
        polydraw.polygon(self.coords[:self.n,:].flatten().tolist(),
                         fill=tuple(self.color))
        return(poly)

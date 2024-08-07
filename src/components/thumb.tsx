import { useEffect, useState, useRef } from "react";

export const useIntersection = (element, rootMargin) => {
  const [isVisible, setState] = useState(false);

  useEffect(() => {
    const current = element?.current;
    const observer = new IntersectionObserver(
      ([entry]) => {
        setState(entry.isIntersecting);
      },
      { rootMargin }
    );
    current && observer?.observe(current);

    return () => current && observer.unobserve(current);
  }, []);

  return isVisible;
};

export const DummyElement = ({ callbackFn }) => {
    const triggerRef = useRef(null);
    const isVisible = useIntersection(triggerRef, "0px");
  
    useEffect(() => {
      if (isVisible) {
        callbackFn(); // Trigger a function when the div is visible on view port
      }
    }, [callbackFn, isVisible]);
  
    return (<div ref={triggerRef}></div>);
};